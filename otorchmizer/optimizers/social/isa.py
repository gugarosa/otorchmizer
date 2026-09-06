# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Interactive Search Algorithm.

References:
    A. Mortazavi, V. Toğan and A. Nuhoğlu.
    Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
    Engineering Applications of Artificial Intelligence (2018).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class ISA(Optimizer):
    """Interactive Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ISA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.w = 0.7
        self.tau = 0.3
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        n = population.n_agents
        if n < 2:
            raise ValueError("`population.n_agents` must be at least 2 for ISA.")

        shape = (n, population.n_variables, population.n_dimensions)
        self.local_position = population.positions.new_zeros(shape)
        self.velocity = population.positions.new_zeros(shape)
        self.local_fitness = population.fitness.new_full((n,), torch.inf)

    def evaluate(self, population, function) -> None:
        """Evaluate a population and update optimizer-specific best state.

        Args:
            population: Population whose tensors define the optimizer state.
            function: Objective function used to score the population.

        """

        fitness = function(population.positions)
        improved = fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved]
        self.local_fitness[improved] = fitness[improved]
        population.fitness = fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ISA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        if not torch.isfinite(pop.fitness).all():
            raise ValueError("`population.fitness` must contain finite values for ISA.")

        scale = pop.fitness.abs().max()
        scaled_fitness = torch.zeros_like(pop.fitness) if scale == 0 else pop.fitness / scale
        shifted = scaled_fitness - scaled_fitness.min()
        total = shifted.sum()
        weights = torch.full_like(shifted, 1 / n) if total == 0 else shifted / total
        weighted_position = (weights.view(n, 1, 1) * pop.positions).sum(dim=0)
        weighted_position = weighted_position.clamp(min=pop.lb, max=pop.ub)
        weighted_fitness = ctx.function(weighted_position.unsqueeze(0))[0]
        prior_local_position = self.local_position.clone()
        prior_local_fitness = self.local_fitness.clone()

        for i in range(n):
            r1 = torch.rand((), device=device, dtype=pop.dtype).item()
            idx = torch.randint(0, n, (1,), device=device).item()
            while idx == i:
                idx = torch.randint(0, n, (1,), device=device).item()

            if r1 >= self.tau:
                phi3 = torch.rand((), device=device, dtype=pop.dtype)
                phi2 = 2 * torch.rand((), device=device, dtype=pop.dtype)
                phi1 = -(phi2 + phi3) * torch.rand((), device=device, dtype=pop.dtype)

                self.velocity[i] = (
                    self.w * self.velocity[i]
                    + phi1 * (prior_local_position[idx] - pop.positions[i])
                    + phi2 * (best.squeeze(0) - prior_local_position[idx])
                    + phi3 * (weighted_position - prior_local_position[idx])
                )
            else:
                r2 = torch.rand((), device=device, dtype=pop.dtype)
                if pop.fitness[i] < pop.fitness[idx]:
                    self.velocity[i] = r2 * (pop.positions[i] - pop.positions[idx])
                else:
                    self.velocity[i] = r2 * (pop.positions[idx] - pop.positions[i])

            pop.positions[i] = pop.positions[i] + self.velocity[i]

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        moved_fitness = ctx.function(pop.positions)
        use_weighted = weighted_fitness < moved_fitness
        selected_fitness = torch.where(use_weighted, weighted_fitness, moved_fitness)
        selected_position = torch.where(use_weighted.view(n, 1, 1), weighted_position, pop.positions)
        improved = selected_fitness < prior_local_fitness
        self.local_position[improved] = selected_position[improved]
        self.local_fitness[improved] = selected_fitness[improved]
        pop.fitness = moved_fitness
        pop.update_best()
        if weighted_fitness < pop.best_fitness:
            pop.best_position = weighted_position.clone()
            pop.best_fitness = weighted_fitness.clone()
