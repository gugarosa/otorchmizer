"""Multi-Verse Optimizer.

References:
    S. Mirjalili, S. M. Mirjalili, and A. Hatamlou.
    Multi-Verse Optimizer: a nature-inspired algorithm for global optimization.
    Neural Computing and Applications (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MVO(Optimizer):
    """Multi-Verse Optimizer.

    White hole, wormhole, and black hole mechanisms.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> MVO.")

        self.WEP_min = 0.2
        self.WEP_max = 1.0
        self.p = 6.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def WEP_min(self) -> float:
        return self._WEP_min

    @WEP_min.setter
    def WEP_min(self, WEP_min: float) -> None:
        if not isinstance(WEP_min, (float, int)):
            raise e.TypeError("`WEP_min` should be a float or integer")
        self._WEP_min = WEP_min

    @property
    def WEP_max(self) -> float:
        return self._WEP_max

    @WEP_max.setter
    def WEP_max(self, WEP_max: float) -> None:
        if not isinstance(WEP_max, (float, int)):
            raise e.TypeError("`WEP_max` should be a float or integer")
        self._WEP_max = WEP_max

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        self._p = p

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration + 1
        T = max(ctx.n_iterations, 1)

        WEP = self.WEP_min + t * (self.WEP_max - self.WEP_min) / T
        TDR = 1 - (t ** (1 / self.p)) / (T ** (1 / self.p))

        # Normalize fitness for roulette
        sorted_idx = torch.argsort(pop.fitness)
        norm_fit = pop.fitness - pop.fitness.min()
        norm_fit = norm_fit / (norm_fit.sum() + 1e-10)

        new_positions = pop.positions.clone()

        for i in range(n):
            for j in range(pop.n_variables):
                r1 = torch.rand(1, device=device).item()

                if r1 < norm_fit[i]:
                    # White hole
                    k = torch.multinomial(1 - norm_fit + 1e-10, 1).item()
                    new_positions[i, j] = pop.positions[k, j]

                r2 = torch.rand(1, device=device).item()
                if r2 < WEP:
                    r3 = torch.rand(1, device=device).item()
                    width = ub.squeeze(0)[j] - lb.squeeze(0)[j]
                    if r3 < 0.5:
                        new_positions[i, j] = best.squeeze(0)[j] + TDR * width * torch.rand(pop.n_dimensions, device=device)
                    else:
                        new_positions[i, j] = best.squeeze(0)[j] - TDR * width * torch.rand(pop.n_dimensions, device=device)

        pop.positions = new_positions.clamp(min=lb, max=ub)
