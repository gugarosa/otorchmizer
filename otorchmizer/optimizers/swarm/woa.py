# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Whale Optimization Algorithm — fully vectorized leader-follower pattern.

References:
    S. Mirjalili and A. Lewis.
    The Whale Optimization Algorithm.
    Advances in Engineering Software (2016).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class WOA(Optimizer):
    """Whale Optimization Algorithm.

    Notes:
        Mimics the bubble-net feeding behavior of humpback whales.
        All position updates are vectorized across the entire population.

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
        self._b = b

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        t = ctx.iteration / max(ctx.n_iterations, 1)

        a = 2.0 - 2.0 * t

        device = pop.device
        n = pop.n_agents

        A = 2.0 * a * torch.rand(n, 1, 1, device=device, dtype=pop.dtype) - a
        C = 2.0 * torch.rand(n, 1, 1, device=device, dtype=pop.dtype)

        spiral_parameter = torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * 2.0 - 1.0
        p = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)

        best = pop.best_position.unsqueeze(0)

        D = torch.abs(C * best - pop.positions)
        encircle = best - A * D

        D_prime = torch.abs(best - pop.positions)
        spiral = D_prime * torch.exp(self.b * spiral_parameter) * torch.cos(2.0 * torch.pi * spiral_parameter) + best

        rand_idx = torch.randint(0, n, (n,), device=device)
        rand_pos = pop.positions[rand_idx]
        D_rand = torch.abs(C * rand_pos - pop.positions)
        explore = rand_pos - A * D_rand

        use_spiral = p >= 0.5
        use_explore = (A.abs() >= 1.0) & (~use_spiral)

        new_positions = torch.where(use_spiral, spiral, torch.where(use_explore, explore, encircle))

        pop.positions = new_positions
