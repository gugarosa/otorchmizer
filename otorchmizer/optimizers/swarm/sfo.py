# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sailfish Optimizer.

References:
    S. Shadravan, H. R. Naji, and V. K. Bardsiri.
    The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
    for solving constrained engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2019).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SFO(Optimizer):
    """Sailfish Optimizer.

    Notes:
        Elite and sardine-based cooperative hunting.
        The v3.1.4 replacement loop only rebound a local agent variable and never consumed prey.
        This implementation applies equation 13 explicitly, removes promoted sardines, and uniformly replenishes
        them so the ``PP``-defined prey pool remains available for subsequent hunting phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.PP = 0.1
        self.A = 4.0
        self.e = 0.001

        super().__init__(params)

    @property
    def PP(self) -> float:
        """Return the initial population proportion."""

        return self._PP

    @PP.setter
    def PP(self, PP: float) -> None:
        if not isinstance(PP, (float, int)):
            raise TypeError("`PP` must be a float or integer.")
        if not 0 < PP <= 1:
            raise ValueError("`PP` must be greater than 0 and at most 1.")
        self._PP = PP

    @property
    def A(self) -> float:
        """Return the attack-power amplitude."""

        return self._A

    @A.setter
    def A(self, A: float) -> None:
        if not isinstance(A, (float, int)):
            raise TypeError("`A` must be a float or integer.")
        if A < 0:
            raise ValueError("`A` must be non-negative.")
        self._A = A

    @property
    def e(self) -> float:
        """Return the attack-power decay."""

        return self._e

    @e.setter
    def e(self, e_val: float) -> None:
        if not isinstance(e_val, (float, int)):
            raise TypeError("`e` must be a float or integer.")
        if e_val < 0:
            raise ValueError("`e` must be non-negative.")
        self._e = e_val

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.n_sailfish = population.n_agents
        self.n_sardines = int(population.n_agents / self.PP)
        lb = population.lb.unsqueeze(0)
        ub = population.ub.unsqueeze(0)
        shape = (self.n_sardines, population.n_variables, population.n_dimensions)
        self.sardine_positions = torch.rand(shape, device=population.device, dtype=population.dtype) * (ub - lb) + lb
        self.sardine_fitness = torch.full(
            (self.n_sardines,), torch.inf, device=population.device, dtype=population.dtype
        )

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        if torch.isinf(self.sardine_fitness).any():
            self.sardine_fitness = fn(self.sardine_positions)
            self._update_archive(pop, self.sardine_positions, self.sardine_fitness)

        best_sardine_idx = self.sardine_fitness.argmin()
        best_sardine = self.sardine_positions[best_sardine_idx].unsqueeze(0)
        fish_best = pop.positions[pop.fitness.argmin()].unsqueeze(0)

        prey_density = 1 - self.n_sailfish / (self.n_sailfish + self.n_sardines)
        lambda_random = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        position_random = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        lambda_i = 2 * lambda_random * prey_density - prey_density
        new_positions = best_sardine - lambda_i * (position_random * (fish_best - best_sardine) / 2 - pop.positions)

        new_positions = new_positions.clamp(min=lb, max=ub)
        new_fitness = fn(new_positions)

        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
        pop.update_best()
        fish_best = pop.positions[pop.fitness.argmin()].unsqueeze(0)

        attack_power = abs(self.A * (1 - 2 * ctx.iteration * self.e))
        if attack_power < 0.5:
            n_selected = int(self.n_sardines * attack_power)
            n_variables = int(pop.n_variables * attack_power)
            selected_sardines = torch.randperm(self.n_sardines, device=device)[:n_selected]
            selected_variables = torch.randperm(pop.n_variables, device=device)[:n_variables]
            if n_selected and n_variables:
                random_factor = torch.rand(n_selected, n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
                current = self.sardine_positions[selected_sardines[:, None], selected_variables]
                self.sardine_positions[selected_sardines[:, None], selected_variables] = random_factor * (
                    fish_best[:, selected_variables] - current + attack_power
                )
        else:
            random_factor = torch.rand(self.n_sardines, 1, 1, device=device, dtype=pop.dtype)
            self.sardine_positions = random_factor * (fish_best - self.sardine_positions + attack_power)

        self.sardine_positions = self.sardine_positions.clamp(min=lb, max=ub)
        self.sardine_fitness = fn(self.sardine_positions)
        self._update_archive(pop, self.sardine_positions, self.sardine_fitness)

        all_positions = torch.cat((pop.positions, self.sardine_positions), dim=0)
        all_fitness = torch.cat((pop.fitness, self.sardine_fitness), dim=0)
        selected = torch.argsort(all_fitness)[:n]
        pop.positions = all_positions[selected]
        pop.fitness = all_fitness[selected]

        consumed = selected[selected >= n] - n
        keep = torch.ones(self.n_sardines, device=device, dtype=torch.bool)
        keep[consumed] = False
        self.sardine_positions = self.sardine_positions[keep]
        self.sardine_fitness = self.sardine_fitness[keep]
        n_consumed = consumed.numel()
        if n_consumed:
            replenished = torch.rand_like(pop.positions[:n_consumed]) * (ub - lb) + lb
            replenished_fitness = fn(replenished)
            self.sardine_positions = torch.cat((self.sardine_positions, replenished), dim=0)
            self.sardine_fitness = torch.cat((self.sardine_fitness, replenished_fitness), dim=0)
            self._update_archive(pop, replenished, replenished_fitness)

        pop.update_best()

    @staticmethod
    def _update_archive(population, positions: torch.Tensor, fitness: torch.Tensor) -> None:
        best_idx = fitness.argmin()
        if fitness[best_idx] < population.best_fitness:
            population.best_fitness = fitness[best_idx].clone()
            population.best_position = positions[best_idx].clone()
