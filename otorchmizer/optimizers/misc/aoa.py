"""Arithmetic Optimization Algorithm.

References:
    L. Abualigah et al.
    The Arithmetic Optimization Algorithm.
    Computer Methods in Applied Mechanics and Engineering (2021).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AOA(Optimizer):
    """Arithmetic Optimization Algorithm.

    Division/multiplication (exploration) and subtraction/addition (exploitation).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> AOA.")

        self.a_min = 0.2
        self.a_max = 1.0
        self.alpha = 5.0
        self.mu = 0.499

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def a_min(self) -> float:
        return self._a_min

    @a_min.setter
    def a_min(self, a_min: float) -> None:
        if not isinstance(a_min, (float, int)):
            raise e.TypeError("`a_min` should be a float or integer")
        self._a_min = a_min

    @property
    def a_max(self) -> float:
        return self._a_max

    @a_max.setter
    def a_max(self, a_max: float) -> None:
        if not isinstance(a_max, (float, int)):
            raise e.TypeError("`a_max` should be a float or integer")
        self._a_max = a_max

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        if not isinstance(mu, (float, int)):
            raise e.TypeError("`mu` should be a float or integer")
        self._mu = mu

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        MOA = self.a_min + t * ((self.a_max - self.a_min) / T)
        MOP = 1 - ((t + 1) ** (1 / self.alpha)) / (T ** (1 / self.alpha))

        search_partition = (ub - lb) * self.mu + lb

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        r3 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)

        # Exploration (division/multiplication)
        div_update = best / (MOP + c.EPSILON) * search_partition
        mul_update = best * MOP * search_partition
        explore = torch.where(r2 > 0.5, div_update, mul_update)

        # Exploitation (subtraction/addition)
        sub_update = best - MOP * search_partition
        add_update = best + MOP * search_partition
        exploit = torch.where(r3 > 0.5, sub_update, add_update)

        pop.positions = torch.where(r1 > MOA, explore, exploit)
        pop.positions = pop.positions.clamp(min=lb, max=ub)
