"""Sine Cosine Algorithm.

References:
    S. Mirjalili.
    SCA: a Sine Cosine Algorithm for solving optimization problems.
    Knowledge-Based Systems (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SCA(Optimizer):
    """Sine Cosine Algorithm.

    Fully vectorized sine/cosine oscillation toward best position.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SCA.")

        self.r_min = 0.0
        self.a = 3.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def r_min(self) -> float:
        return self._r_min

    @r_min.setter
    def r_min(self, r_min: float) -> None:
        if not isinstance(r_min, (float, int)):
            raise e.TypeError("`r_min` should be a float or integer")
        self._r_min = r_min

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` should be a float or integer")
        self._a = a

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Linearly decreasing r1
        r1 = self.a - t * (self.a - self.r_min)

        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) * 2 * torch.pi
        r3 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) * 2
        r4 = torch.rand(n, device=device)

        use_sine = (r4 < 0.5).view(n, 1, 1)

        sine_update = pop.positions + r1 * torch.sin(r2) * torch.abs(r3 * best - pop.positions)
        cosine_update = pop.positions + r1 * torch.cos(r2) * torch.abs(r3 * best - pop.positions)

        pop.positions = torch.where(use_sine, sine_update, cosine_update)
