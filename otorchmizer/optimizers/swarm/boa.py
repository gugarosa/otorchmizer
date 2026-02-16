"""Butterfly Optimization Algorithm.

References:
    S. Arora and S. Singh.
    Butterfly optimization algorithm: a novel approach for global optimization.
    Soft Computing (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BOA(Optimizer):
    """Butterfly Optimization Algorithm.

    Fragrance-based search with global and local phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BOA.")

        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if not isinstance(c, (float, int)):
            raise e.TypeError("`c` should be a float or integer")
        self._c = c

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` should be a float or integer")
        self._a = a

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        if not 0 <= p <= 1:
            raise e.ValueError("`p` should be between 0 and 1")
        self._p = p

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        # Calculate fragrance
        fragrance = self.c * pop.fitness.abs() ** self.a  # (n,)
        f = fragrance.view(n, 1, 1)

        r = torch.rand(n, 1, 1, device=device)
        prob = torch.rand(n, device=device)

        # Global search: move toward best
        global_pos = pop.positions + (r ** 2) * (best - pop.positions) * f

        # Local search: move toward random neighbor
        j = torch.randint(0, n, (n,), device=device)
        k = torch.randint(0, n, (n,), device=device)
        local_pos = pop.positions + (r ** 2) * (pop.positions[j] - pop.positions[k]) * f

        use_global = (prob < self.p).view(n, 1, 1)
        pop.positions = torch.where(use_global, global_pos, local_pos)
