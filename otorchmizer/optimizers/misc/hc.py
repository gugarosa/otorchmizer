# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Hill Climbing.

References:
    S. Skiena. The Algorithm Design Manual (2010).
"""

from __future__ import annotations

from typing import Any

import otorchmizer.math.random as r
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class HC(Optimizer):
    """Hill Climbing optimizer.

    Notes:
        Perturbs each agent with Gaussian noise before the outer loop evaluates and tracks the best candidate.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the HC optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.r_mean = 0.0
        self.r_var = 0.1

        super().__init__(params)

    @property
    def r_mean(self) -> float:
        """Return the noise mean.

        Returns:
            float: Current noise mean.

        """

        return self._r_mean

    @r_mean.setter
    def r_mean(self, r_mean: float) -> None:
        """Set the noise mean.

        Args:
            r_mean: New value for the noise mean.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(r_mean, (float, int)):
            raise TypeError("`r_mean` must be a float or integer.")
        self._r_mean = r_mean

    @property
    def r_var(self) -> float:
        """Return the noise variance.

        Returns:
            float: Current noise variance.

        """

        return self._r_var

    @r_var.setter
    def r_var(self, r_var: float) -> None:
        """Set the noise variance.

        Args:
            r_var: New value for the noise variance.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(r_var, (float, int)):
            raise TypeError("`r_var` must be a float or integer.")
        if r_var < 0:
            raise ValueError("`r_var` must be non-negative.")
        self._r_var = r_var

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one HC step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population

        noise = r.generate_gaussian_random_number(
            mean=self.r_mean,
            variance=self.r_var,
            size=pop.positions.shape,
            device=pop.device,
        )

        pop.positions = pop.positions + noise
