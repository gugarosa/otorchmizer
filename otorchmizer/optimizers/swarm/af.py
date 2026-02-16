"""Artificial Flora.

References:
    L. Cheng et al. Artificial flora (AF) optimization algorithm.
    Applied Sciences (2018).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AF(Optimizer):
    """Artificial Flora optimizer.

    Simulates the spreading and competition of plant species.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> AF.")

        self.m = 5
        self.Q = 0.75
        self.g = 0.4

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def m(self) -> int:
        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if not isinstance(m, int):
            raise e.TypeError("`m` should be an integer")
        if m <= 0:
            raise e.ValueError("`m` should be > 0")
        self._m = m

    @property
    def Q(self) -> float:
        return self._Q

    @Q.setter
    def Q(self, Q: float) -> None:
        if not isinstance(Q, (float, int)):
            raise e.TypeError("`Q` should be a float or integer")
        self._Q = Q

    @property
    def g(self) -> float:
        return self._g

    @g.setter
    def g(self, g: float) -> None:
        if not isinstance(g, (float, int)):
            raise e.TypeError("`g` should be a float or integer")
        self._g = g

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.branch = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for i in range(n):
            # Generate m seedlings per plant
            best_seed_pos = None
            best_seed_fit = pop.fitness[i]

            for _ in range(self.m):
                delta = self.g * torch.randn_like(pop.positions[i]) + self.Q * self.branch[i]
                seed_pos = pop.positions[i] + delta
                seed_pos = seed_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                seed_fit = fn(seed_pos.unsqueeze(0))[0]

                if seed_fit < best_seed_fit:
                    best_seed_fit = seed_fit
                    best_seed_pos = seed_pos.clone()

            if best_seed_pos is not None:
                self.branch[i] = best_seed_pos - pop.positions[i]
                pop.positions[i] = best_seed_pos
                pop.fitness[i] = best_seed_fit
            else:
                self.branch[i] = torch.zeros_like(self.branch[i])
