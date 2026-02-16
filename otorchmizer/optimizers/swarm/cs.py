"""Cuckoo Search.

References:
    X.-S. Yang and S. Deb.
    Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing (2009).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class CS(Optimizer):
    """Cuckoo Search optimizer.

    Vectorized Lévy flight exploration with fraction of worst nests abandoned.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> CS.")

        self.alpha = 1.0
        self.beta = 1.5
        self.p = 0.2

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
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        self._beta = beta

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
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Lévy flight step
        levy = d.generate_levy_distribution(
            beta=self.beta,
            size=pop.positions.shape,
            device=device,
        )
        step_size = self.alpha * levy * (pop.positions - best)
        new_positions = pop.positions + step_size
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        # Abandon worst nests
        abandon = torch.rand(n, device=device) < self.p
        if abandon.any():
            # Random permutation to create new nests
            perm1 = torch.randperm(n, device=device)
            perm2 = torch.randperm(n, device=device)
            step = torch.rand(n, 1, 1, device=device) * (pop.positions[perm1] - pop.positions[perm2])
            new_pos2 = pop.positions + step
            new_pos2 = new_pos2.clamp(min=lb, max=ub)

            new_fit2 = fn(new_pos2)
            replace = abandon & (new_fit2 < pop.fitness)
            pop.positions[replace] = new_pos2[replace]
            pop.fitness[replace] = new_fit2[replace]
