# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Wind Driven Optimization.

References:
    Z. Bayraktar et al.
    The wind driven optimization technique and its application in
    electromagnetics.
    IEEE Transactions on Antennas and Propagation (2013).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WDO(Optimizer):
    """Wind Driven Optimization.

    Notes:
        Models air-parcel movement using pressure, Coriolis, gravity, and friction terms.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the WDO optimizer.

        Args:
            params: Algorithm parameter overrides.

        Raises:
            TypeError: A force or velocity coefficient is not numeric.
            ValueError: A coefficient is negative or friction is outside [0, 1].

        """

        logger.info("Overriding class: Optimizer -> WDO.")

        self.v_max = 0.3
        self.alpha = 0.8
        self.g = 0.6
        self.c = 1.0
        self.RT = 1.5

        super().__init__(params)
        for name in ("g", "c", "RT"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)):
                raise e.TypeError(f"`{name}` must be a float or integer.")
            if value < 0:
                raise e.ValueError(f"`{name}` must be nonnegative.")

        logger.info("Class overrided.")

    @property
    def v_max(self) -> float:
        """Return the maximum velocity.

        Returns:
            float: Current maximum velocity.

        """

        return self._v_max

    @v_max.setter
    def v_max(self, v_max: float) -> None:
        """Set the maximum velocity.

        Args:
            v_max: New value for the maximum velocity.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the maximum velocity is negative.

        """

        if not isinstance(v_max, (float, int)):
            raise e.TypeError("`v_max` must be a float or integer.")
        if v_max < 0:
            raise e.ValueError("`v_max` must be nonnegative.")
        self._v_max = v_max

    @property
    def alpha(self) -> float:
        """Return the alpha coefficient.

        Returns:
            float: Current alpha coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set the alpha coefficient.

        Args:
            alpha: New value for the alpha coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If friction is outside [0, 1].

        """

        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        if not 0 <= alpha <= 1:
            raise e.ValueError("`alpha` must be between 0 and 1.")
        self._alpha = alpha

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = population.positions.new_zeros(shape)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one WDO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

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
                + self.c * self.velocity[idx] / (idx + 1)
            )

            self.velocity[i] = new_vel.clamp(min=-self.v_max, max=self.v_max)

        pop.positions = pop.positions + self.velocity
