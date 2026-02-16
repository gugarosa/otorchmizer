"""Whale Optimization Algorithm — fully vectorized leader-follower pattern.

References:
    S. Mirjalili and A. Lewis.
    The Whale Optimization Algorithm.
    Advances in Engineering Software (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WOA(Optimizer):
    """Whale Optimization Algorithm.

    Mimics the bubble-net feeding behavior of humpback whales.
    All position updates are vectorized across the entire population.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> WOA.")

        self.b = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, b: float) -> None:
        if not isinstance(b, (float, int)):
            raise e.TypeError("`b` should be a float or integer")
        self._b = b

    def update(self, ctx: UpdateContext) -> None:
        """Vectorized WOA update — spiral + encircling prey."""

        pop = ctx.space.population
        t = ctx.iteration / max(ctx.n_iterations, 1)

        a = 2.0 - 2.0 * t  # linearly decreasing from 2 to 0

        shape = pop.positions.shape
        device = pop.device
        n = pop.n_agents

        r = torch.rand(n, 1, 1, device=device)
        A = 2.0 * a * torch.rand(n, 1, 1, device=device) - a
        C = 2.0 * torch.rand(n, 1, 1, device=device)

        l = torch.rand(n, 1, 1, device=device) * 2.0 - 1.0  # [-1, 1]
        p = torch.rand(n, 1, 1, device=device)

        best = pop.best_position.unsqueeze(0)  # (1, n_vars, n_dims)

        # Encircling prey
        D = torch.abs(C * best - pop.positions)
        encircle = best - A * D

        # Spiral update
        D_prime = torch.abs(best - pop.positions)
        spiral = D_prime * torch.exp(self.b * l) * torch.cos(2.0 * torch.pi * l) + best

        # Random search (exploration): use random agent when |A| >= 1
        rand_idx = torch.randint(0, n, (n,), device=device)
        rand_pos = pop.positions[rand_idx]
        D_rand = torch.abs(C * rand_pos - pop.positions)
        explore = rand_pos - A * D_rand

        # Select behavior: p < 0.5 → encircle/explore, else → spiral
        use_spiral = p >= 0.5
        use_explore = (A.abs() >= 1.0) & (~use_spiral)

        new_positions = torch.where(use_spiral, spiral,
                        torch.where(use_explore, explore, encircle))

        pop.positions = new_positions
