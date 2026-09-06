# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Arithmetic Optimization Algorithm.

References:
    L. Abualigah et al.
    The Arithmetic Optimization Algorithm.
    Computer Methods in Applied Mechanics and Engineering (2021).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class AOA(Optimizer):
    """Arithmetic Optimization Algorithm.

    Notes:
        Uses division and multiplication for exploration, then subtraction and addition for exploitation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the AOA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.a_min = 0.2
        self.a_max = 1.0
        self.alpha = 5.0
        self.mu = 0.499

        super().__init__(params)

    @property
    def a_min(self) -> float:
        """Return the minimum accelerated function.

        Returns:
            float: Current minimum accelerated function.

        """

        return self._a_min

    @a_min.setter
    def a_min(self, a_min: float) -> None:
        """Set the minimum accelerated function.

        Args:
            a_min: New value for the minimum accelerated function.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(a_min, (float, int)):
            raise TypeError("`a_min` must be a float or integer.")
        self._a_min = a_min

    @property
    def a_max(self) -> float:
        """Return the maximum accelerated function.

        Returns:
            float: Current maximum accelerated function.

        """

        return self._a_max

    @a_max.setter
    def a_max(self, a_max: float) -> None:
        """Set the maximum accelerated function.

        Args:
            a_max: New value for the maximum accelerated function.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(a_max, (float, int)):
            raise TypeError("`a_max` must be a float or integer.")
        self._a_max = a_max

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

        """

        if not isinstance(alpha, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        self._alpha = alpha

    @property
    def mu(self) -> float:
        """Return the control coefficient.

        Returns:
            float: Current control coefficient.

        """

        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        """Set the control coefficient.

        Args:
            mu: New value for the control coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(mu, (float, int)):
            raise TypeError("`mu` must be a float or integer.")
        self._mu = mu

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one AOA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

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
