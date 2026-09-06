# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Artificial Flora.

References:
    L. Cheng, W. Xue-han, and Y. Wang.
    Artificial flora optimization algorithm.
    Applied Sciences (2018).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class AF(Optimizer):
    """Artificial Flora optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c1 = 0.75
        self.c2 = 1.25
        self.m = 10
        self.Q = 0.75

        super().__init__(params)

    @property
    def c1(self) -> float:
        """Return the grandparent-distance coefficient."""

        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        self._c1 = self._validate_nonnegative("c1", c1)

    @property
    def c2(self) -> float:
        """Return the parent-distance coefficient."""

        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        self._c2 = self._validate_nonnegative("c2", c2)

    @property
    def m(self) -> int:
        """Return the offspring count per flora."""

        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if not isinstance(m, int):
            raise TypeError("`m` must be an integer.")
        if m <= 0:
            raise ValueError("`m` must be positive.")
        self._m = m

    @property
    def Q(self) -> float:
        """Return the offspring-selection scale."""

        return self._Q

    @Q.setter
    def Q(self, Q: float) -> None:
        if not isinstance(Q, (float, int)):
            raise TypeError("`Q` must be a float or integer.")
        if not 0 <= Q <= 1:
            raise ValueError("`Q` must be between 0 and 1.")
        self._Q = Q

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(f"`{name}` must be a float or integer.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"`{name}` must be finite and non-negative.")
        return value

    def compile(self, population) -> None:
        """Initialize parent and grandparent dispersal distances.

        Args:
            population: Population that defines state shape, device, and dtype.

        """

        self.p_distance = torch.rand(
            population.n_agents,
            device=population.device,
            dtype=population.dtype,
        )
        self.g_distance = torch.rand(
            population.n_agents,
            device=population.device,
            dtype=population.dtype,
        )

    def update(self, ctx: UpdateContext) -> None:
        """Advance the flora population by one propagation and selection cycle.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        order = torch.argsort(pop.fitness)
        positions = pop.positions[order]
        parent_distance = self.p_distance[order]
        grandparent_distance = self.g_distance[order]
        random_shape = (pop.n_agents, self.m)
        distance = (
            grandparent_distance[:, None] * torch.rand(random_shape, device=pop.device, dtype=pop.dtype) * self.c1
            + parent_distance[:, None] * torch.rand(random_shape, device=pop.device, dtype=pop.dtype) * self.c2
        )
        offspring = (
            positions[:, None]
            + torch.randn(
                pop.n_agents,
                self.m,
                pop.n_variables,
                pop.n_dimensions,
                device=pop.device,
                dtype=pop.dtype,
            )
            * distance[:, :, None, None]
        )
        offspring = offspring.clamp(
            min=pop.lb.unsqueeze(0).unsqueeze(0),
            max=pop.ub.unsqueeze(0).unsqueeze(0),
        )
        flat_offspring = offspring.reshape(-1, pop.n_variables, pop.n_dimensions)
        offspring_fitness = ctx.function(flat_offspring)
        best_offspring = offspring_fitness.argmin()
        if offspring_fitness[best_offspring] < pop.best_fitness:
            pop.best_fitness = offspring_fitness[best_offspring].clone()
            pop.best_position = flat_offspring[best_offspring].clone()

        worst_scale = pop.fitness[order[-1]].abs().clamp_min(torch.finfo(pop.dtype).tiny)
        probability = self.Q * torch.sqrt((offspring_fitness.abs() / worst_scale).clamp_min(0))
        selected = torch.rand_like(probability) < probability.clamp(max=1)
        selected_indices = selected.nonzero(as_tuple=False).flatten()
        if selected_indices.numel() < pop.n_agents:
            ranked = torch.argsort(offspring_fitness)
            unused = ranked[~torch.isin(ranked, selected_indices)]
            selected_indices = torch.cat((selected_indices, unused[: pop.n_agents - selected_indices.numel()]))
        elif selected_indices.numel() > pop.n_agents:
            permutation = torch.randperm(selected_indices.numel(), device=pop.device)
            selected_indices = selected_indices[permutation[: pop.n_agents]]

        parent_indices = torch.div(selected_indices, self.m, rounding_mode="floor")
        pop.positions = flat_offspring[selected_indices]
        pop.fitness = offspring_fitness[selected_indices]
        self.g_distance = parent_distance[parent_indices]
        displacement = positions[parent_indices] - pop.positions
        self.p_distance = displacement.square().reshape(pop.n_agents, -1).mean(dim=1).sqrt()
        pop.update_best()
