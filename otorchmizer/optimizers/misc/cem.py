# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Cross-Entropy Method.

References:
    R. Y. Rubinstein.
    Optimization of computer simulation models with rare events.
    European Journal of Operational Research (1997).
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class CEM(Optimizer):
    """Cross-Entropy Method.

    Notes:
        Samples a parameterized distribution and adapts it from elite candidates.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the CEM optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_updates = 5
        self.alpha = 0.7

        super().__init__(params)

    @property
    def n_updates(self) -> int:
        """Return the number of elite updates.

        Returns:
            int: Current number of elite updates.

        """

        return self._n_updates

    @n_updates.setter
    def n_updates(self, n_updates: int) -> None:
        """Set the number of elite updates.

        Args:
            n_updates: New value for the number of elite updates.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(n_updates, Integral):
            raise TypeError("`n_updates` must be an integer.")
        if n_updates <= 0:
            raise ValueError("`n_updates` must be positive.")
        self._n_updates = n_updates

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
            ValueError: If the supplied value is negative or non-finite.

        """

        if not isinstance(alpha, Real):
            raise TypeError("`alpha` must be a real number.")
        if not isfinite(alpha) or alpha < 0:
            raise ValueError("`alpha` must be finite and non-negative.")
        self._alpha = alpha

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        lb = population.lb.expand(-1, population.n_dimensions)
        ub = population.ub.expand(-1, population.n_dimensions)

        self.mean = torch.rand_like(population.positions[0]) * (ub - lb) + lb
        self.std = (ub - lb).clone()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one CEM step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Sample new positions from current distribution
        pop.positions = torch.randn_like(pop.positions) * self.std.unsqueeze(0) + self.mean.unsqueeze(0)

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        pop.fitness = fn(pop.positions)

        # Sort and select elite
        sorted_idx = torch.argsort(pop.fitness)
        n_elite = min(self.n_updates, n)
        elite = pop.positions[sorted_idx[:n_elite]]

        # Update mean and std with exponential moving average
        elite_mean = elite.mean(dim=0)
        elite_std = ((elite - elite_mean.unsqueeze(0)) ** 2).mean(dim=0).sqrt()

        self.mean = self.alpha * self.mean + (1 - self.alpha) * elite_mean
        self.std = self.alpha * self.std + (1 - self.alpha) * elite_std
