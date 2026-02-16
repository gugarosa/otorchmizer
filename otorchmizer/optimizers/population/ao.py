"""Aquila Optimizer.

References:
    L. Abualigah et al.
    Aquila Optimizer: A novel meta-heuristic optimization algorithm.
    Computers & Industrial Engineering (2021).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AO(Optimizer):
    """Aquila Optimizer.

    Four hunting strategies: high/low soar, slow descent, walk & grab.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> AO.")

        self.alpha = 0.1
        self.delta = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, delta: float) -> None:
        if not isinstance(delta, (float, int)):
            raise e.TypeError("`delta` should be a float or integer")
        self._delta = delta

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        avg = pop.positions.mean(dim=0, keepdim=True)

        r1 = torch.rand(n, device=device)
        r2 = torch.rand(n, 1, 1, device=device)

        if t <= 2.0 / 3:
            # Exploration
            use_strategy1 = r1 < 0.5
            s1 = use_strategy1.view(n, 1, 1)

            # Strategy 1: Expanded exploration
            pos1 = best * (1 - t) + (avg - best * r2)

            # Strategy 2: Narrowing exploration with Lévy flight
            levy = d.generate_levy_distribution(beta=1.5, size=pop.positions.shape, device=device)
            j = torch.randint(0, n, (n,), device=device)
            pos2 = best * levy + pop.positions[j] + (torch.rand(n, 1, 1, device=device) * 2 - 1) * r2

            pop.positions = torch.where(s1, pos1, pos2)
        else:
            # Exploitation
            use_strategy3 = r1 < 0.5
            s3 = use_strategy3.view(n, 1, 1)

            # Strategy 3: Expanded exploitation
            pos3 = (best - avg) * self.alpha - r2 + ((ub - lb) * r2 + lb) * self.delta

            # Strategy 4: Narrowing exploitation with Lévy flight
            G1 = 2 * r2 - 1
            G2 = 2 * (1 - t)
            QF = t ** (G1 / ((1 - ctx.n_iterations) ** 2 + 1e-10))
            levy = d.generate_levy_distribution(beta=1.5, size=pop.positions.shape, device=device)
            pos4 = QF * best - (G1 * pop.positions * r2) - G2 * levy + r2 * G1

            pop.positions = torch.where(s3, pos3, pos4)

        pop.positions = pop.positions.clamp(min=lb, max=ub)
