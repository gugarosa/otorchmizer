"""Grasshopper Optimization Algorithm.

References:
    S. Saremi, S. Mirjalili, and A. Lewis.
    Grasshopper Optimisation Algorithm: Theory and application.
    Advances in Engineering Software (2017).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GOA(Optimizer):
    """Grasshopper Optimization Algorithm.

    Social interaction forces with pairwise distance computation.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> GOA.")

        self.c_min = 0.00004
        self.c_max = 1.0
        self.f = 0.5
        self.l = 1.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def c_min(self) -> float:
        return self._c_min

    @c_min.setter
    def c_min(self, c_min: float) -> None:
        if not isinstance(c_min, (float, int)):
            raise e.TypeError("`c_min` should be a float or integer")
        self._c_min = c_min

    @property
    def c_max(self) -> float:
        return self._c_max

    @c_max.setter
    def c_max(self, c_max: float) -> None:
        if not isinstance(c_max, (float, int)):
            raise e.TypeError("`c_max` should be a float or integer")
        self._c_max = c_max

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
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        c = self.c_max - t * (self.c_max - self.c_min)

        # Pairwise distances
        flat = pop.positions.reshape(n, -1)  # (n, n_vars * n_dims)
        dist_matrix = torch.cdist(flat, flat)  # (n, n)
        dist_matrix = dist_matrix.clamp(min=1e-10)

        # Social interaction: s(r) = f * exp(-r/l) - exp(-r)
        s = self.f * torch.exp(-dist_matrix / self.l) - torch.exp(-dist_matrix)
        s.fill_diagonal_(0)

        # Direction vectors (normalized)
        diff = pop.positions.unsqueeze(1) - pop.positions.unsqueeze(0)  # (n, n, v, d)
        norm = dist_matrix.unsqueeze(-1).unsqueeze(-1)
        direction = diff / norm

        # Social force
        force = (c * s.unsqueeze(-1).unsqueeze(-1) * direction).sum(dim=1)

        # Width for normalization
        width = (ub - lb).clamp(min=1e-10)

        pop.positions = c * force / width + best
        pop.positions = pop.positions.clamp(min=lb, max=ub)
