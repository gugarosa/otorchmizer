"""Hill Climbing.

References:
    S. Skiena. The Algorithm Design Manual (2010).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import otorchmizer.math.random as r
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class HC(Optimizer):
    """Hill Climbing optimizer.

    Each agent takes a random step (Gaussian noise added to position).
    The outer evaluate() loop handles fitness comparison and best tracking.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> HC.")

        self.r_mean = 0.0
        self.r_var = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def r_mean(self) -> float:
        return self._r_mean

    @r_mean.setter
    def r_mean(self, r_mean: float) -> None:
        if not isinstance(r_mean, (float, int)):
            raise e.TypeError("`r_mean` should be a float or integer")
        self._r_mean = r_mean

    @property
    def r_var(self) -> float:
        return self._r_var

    @r_var.setter
    def r_var(self, r_var: float) -> None:
        if not isinstance(r_var, (float, int)):
            raise e.TypeError("`r_var` should be a float or integer")
        if r_var < 0:
            raise e.ValueError("`r_var` should be >= 0")
        self._r_var = r_var

    def update(self, ctx: UpdateContext) -> None:
        """Vectorized hill climbing: adds Gaussian noise to all agents (p. 252).

        The outer loop's evaluate() handles fitness comparison and best tracking.
        """

        pop = ctx.space.population

        noise = r.generate_gaussian_random_number(
            mean=self.r_mean,
            variance=self.r_var,
            size=pop.positions.shape,
            device=pop.device,
        )

        pop.positions = pop.positions + noise
