# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Satin Bowerbird Optimizer.

References:
    S. H. S. Moosavi and V. K. Bardsiri.
    Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS.
    Engineering Applications of Artificial Intelligence (2017).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SBO(Optimizer):
    """Satin Bowerbird Optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 0.9
        self.p_mutation = 0.05
        self.z = 0.02

        super().__init__(params)

    @property
    def alpha(self) -> float:
        """Return the attraction coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = self._validate_nonnegative("alpha", alpha)

    @property
    def p_mutation(self) -> float:
        """Return the per-variable mutation probability."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("`p_mutation` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`p_mutation` must be between 0 and 1.")
        self._p_mutation = value

    @property
    def z(self) -> float:
        """Return the mutation-scale coefficient."""

        return self._z

    @z.setter
    def z(self, value: float) -> None:
        self._z = self._validate_nonnegative("z", value)

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(f"`{name}` must be a float or integer.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"`{name}` must be finite and non-negative.")
        return value

    def compile(self, population) -> None:
        """Initialize per-variable Gaussian mutation scales.

        Args:
            population: Population that defines mutation bounds and dtype.

        """

        self.sigma = self.z * (population.ub - population.lb)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one attraction and mutation cycle.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        transformed = torch.where(
            pop.fitness >= 0,
            1 / (1 + pop.fitness),
            1 + pop.fitness.abs(),
        )
        probabilities = transformed / transformed.sum()
        partners = torch.multinomial(probabilities, pop.n_agents, replacement=True)
        attraction = self.alpha / (1 + probabilities[partners])
        candidates = pop.positions + attraction[:, None, None] * (
            (pop.positions[partners] + pop.best_position.unsqueeze(0)) / 2 - pop.positions
        )

        mutation = (
            torch.rand(
                pop.n_agents,
                pop.n_variables,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            < self.p_mutation
        )
        noise = torch.randn(
            pop.n_agents,
            pop.n_variables,
            1,
            device=pop.device,
            dtype=pop.dtype,
        ) * self.sigma.unsqueeze(0)
        candidates = torch.where(mutation, candidates + noise, candidates)
        pop.positions = candidates.clamp(min=pop.lb.unsqueeze(0), max=pop.ub.unsqueeze(0))
        pop.fitness = ctx.function(pop.positions)
        pop.update_best()
