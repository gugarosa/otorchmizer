"""Wind Driven Optimization.

References:
    Z. Bayraktar et al.
    The wind driven optimization technique and its application in
    electromagnetics.
    IEEE Transactions on Antennas and Propagation (2013).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WDO(Optimizer):
    """Wind Driven Optimization.

    Pressure, Coriolis, gravity, and friction-based air parcel movement.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> WDO.")

        self.v_max = 0.3
        self.alpha = 0.8
        self.g = 0.6
        self.c_val = 1.0
        self.RT = 1.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def v_max(self) -> float:
        return self._v_max

    @v_max.setter
    def v_max(self, v_max: float) -> None:
        if not isinstance(v_max, (float, int)):
            raise e.TypeError("`v_max` should be a float or integer")
        self._v_max = v_max

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)

        for i in range(n):
            idx = torch.randint(0, n, (1,), device=device).item()

            # Pressure, gravity, friction, Coriolis
            new_vel = (
                (1 - self.alpha) * self.velocity[i]
                - self.g * pop.positions[i]
                + self.RT * abs(1.0 / (idx + 1) - 1) * (best.squeeze(0) - pop.positions[i])
                + self.c_val * self.velocity[idx] / (idx + 1)
            )

            self.velocity[i] = new_vel.clamp(min=-self.v_max, max=self.v_max)

        pop.positions = pop.positions + self.velocity
