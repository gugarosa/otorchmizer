"""Emperor Penguin Optimizer.

References:
    G. Dhiman and V. Kumar.
    Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
    Knowledge-Based Systems (2018).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class EPO(Optimizer):
    """Emperor Penguin Optimizer.

    Temperature-based huddle dynamics.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> EPO.")

        self.f = 2.0
        self.l = 1.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float) -> None:
        if not isinstance(f, (float, int)):
            raise e.TypeError("`f` should be a float or integer")
        self._f = f

    @property
    def l(self) -> float:
        return self._l

    @l.setter
    def l(self, l: float) -> None:
        if not isinstance(l, (float, int)):
            raise e.TypeError("`l` should be a float or integer")
        self._l = l

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        t = ctx.iteration + 1
        T = max(ctx.n_iterations, 1)

        R = torch.rand(n, 1, 1, device=device)
        T_flag = (R >= 0.5).float()

        # Temperature profile
        T_p = T_flag - T / (t - T + 1e-10)

        # Polygon grid accuracy
        P_grid = torch.abs(best - pop.positions)

        r1 = torch.rand(n, 1, 1, device=device)
        C = torch.rand(n, 1, 1, device=device)

        # Avoidance coefficient
        A = 2 * (T_p + P_grid) * r1 - T_p

        # Social forces
        S = (torch.abs(self.f * torch.exp(torch.tensor(-t / self.l, device=device)) - torch.exp(torch.tensor(-t, dtype=torch.float32, device=device)))) ** 2

        # Distance
        D_ep = torch.abs(S * best - C * pop.positions)

        pop.positions = best - A * D_ep
