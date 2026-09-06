# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Moth-Flame Optimization.

References:
    S. Mirjalili.
    Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems (2015).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class MFO(Optimizer):
    """Moth-Flame Optimization.

    Notes:
        Vectorized moth spiral movement toward sorted flames.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.b = 1.0

        super().__init__(params)

    @property
    def b(self) -> float:
        """Return the spiral coefficient."""

        return self._b

    @b.setter
    def b(self, b: float) -> None:
        if not isinstance(b, (float, int)):
            raise TypeError("`b` must be a float or integer.")
        if not math.isfinite(b):
            raise ValueError("`b` must be finite.")
        self._b = b

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.flames = population.positions.clone()
        self.flame_fitness = population.fitness.clone()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        all_pos = torch.cat([pop.positions, self.flames], dim=0)
        all_fit = torch.cat([pop.fitness, self.flame_fitness], dim=0)
        sorted_idx = torch.argsort(all_fit)[:n]
        self.flames = all_pos[sorted_idx].clone()
        self.flame_fitness = all_fit[sorted_idx].clone()

        t = ctx.iteration / max(ctx.n_iterations, 1)
        n_flames = max(int(n - t * (n - 1)), 1)

        flame_targets = self.flames.clone()

        for i in range(n_flames, n):
            flame_targets[i] = self.flames[n_flames - 1]

        lower = -1 - ctx.iteration / max(ctx.n_iterations, 1)
        t_rand = lower + torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * (1 - lower)
        D = torch.abs(flame_targets - pop.positions)
        pop.positions = D * torch.exp(self.b * t_rand) * torch.cos(2 * torch.pi * t_rand) + flame_targets
