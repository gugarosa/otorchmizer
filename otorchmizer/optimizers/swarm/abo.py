"""African Buffalo Optimization.

References:
    J. Odili and M. Mohmad Kahar.
    Solving the Traveling Salesman's Problem Using the African Buffalo Optimization.
    Computational Intelligence and Neuroscience (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ABO(Optimizer):
    """African Buffalo Optimization.

    Mimics the movement patterns of African buffaloes.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> ABO.")

        self.sunspot_ratio = 0.99
        self.starvation_ratio = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def sunspot_ratio(self) -> float:
        return self._sunspot_ratio

    @sunspot_ratio.setter
    def sunspot_ratio(self, sunspot_ratio: float) -> None:
        if not isinstance(sunspot_ratio, (float, int)):
            raise e.TypeError("`sunspot_ratio` should be a float or integer")
        self._sunspot_ratio = sunspot_ratio

    @property
    def starvation_ratio(self) -> float:
        return self._starvation_ratio

    @starvation_ratio.setter
    def starvation_ratio(self, starvation_ratio: float) -> None:
        if not isinstance(starvation_ratio, (float, int)):
            raise e.TypeError("`starvation_ratio` should be a float or integer")
        self._starvation_ratio = starvation_ratio

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.w1 = torch.zeros(shape, device=population.device)
        self.w2 = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        lp1 = torch.rand(n, 1, 1, device=device)
        lp2 = torch.rand(n, 1, 1, device=device)

        # Update exploitation (w1) and exploration (w2)
        self.w1 = self.w1 + lp1 * (best - pop.positions) + lp2 * (self.w1 - pop.positions)
        self.w2 = self.w2 / 2 + self.w1

        pop.positions = pop.positions + self.w2

        # Starvation reset
        r = torch.rand(n, device=device)
        starving = r < self.starvation_ratio
        if starving.any():
            lb = pop.lb.unsqueeze(0)
            ub = pop.ub.unsqueeze(0)
            n_s = starving.sum().item()
            pop.positions[starving] = torch.rand(
                n_s, pop.n_variables, pop.n_dimensions, device=device
            ) * (ub - lb) + lb
