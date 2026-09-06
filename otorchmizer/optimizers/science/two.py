# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tug Of War Optimization.

References:
    A. Kaveh. Tug of War Optimization.
    Advances in Metaheuristic Algorithms for Optimal Design of Structures (2016).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext


def _nonzero(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


class TWO(Optimizer):
    """Tug of War Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the TWO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.mu_s = 1.0
        self.mu_k = 1.0
        self.delta_t = 1.0
        self.alpha = 0.9
        self.beta = 0.05
        super().__init__(params)

    @property
    def mu_s(self) -> float:
        """Return the static-friction coefficient.

        Returns:
            float: Current static-friction coefficient.

        """

        return self._mu_s

    @mu_s.setter
    def mu_s(self, value: float) -> None:
        """Set the static-friction coefficient.

        Args:
            value: New static-friction coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`mu_s` must be a float or integer.")
        if value < 0:
            raise ValueError("`mu_s` must be non-negative.")
        self._mu_s = float(value)

    @property
    def mu_k(self) -> float:
        """Return the kinetic-friction coefficient.

        Returns:
            float: Current kinetic-friction coefficient.

        """

        return self._mu_k

    @mu_k.setter
    def mu_k(self, value: float) -> None:
        """Set the kinetic-friction coefficient.

        Args:
            value: New kinetic-friction coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`mu_k` must be a float or integer.")
        if value <= 0:
            raise ValueError("`mu_k` must be positive.")
        self._mu_k = float(value)

    @property
    def delta_t(self) -> float:
        """Return the time displacement.

        Returns:
            float: Current time displacement.

        """

        return self._delta_t

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        """Set the time displacement.

        Args:
            value: New time displacement.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`delta_t` must be a float or integer.")
        if value < 0:
            raise ValueError("`delta_t` must be non-negative.")
        self._delta_t = float(value)

    @property
    def alpha(self) -> float:
        """Return the speed coefficient.

        Returns:
            float: Current speed coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the speed coefficient.

        Args:
            value: New speed coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the canonical range.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        if not 0.9 <= value <= 1:
            raise ValueError("`alpha` must be between 0.9 and 1.")
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        """Return the random-displacement scale.

        Returns:
            float: Current random-displacement scale.

        """

        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the random-displacement scale.

        Args:
            value: New random-displacement scale.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the canonical range.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        if not 0 < value <= 1:
            raise ValueError("`beta` must be greater than 0 and at most 1.")
        self._beta = float(value)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one TWO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        t = ctx.iteration + 1

        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        weights = (pop.fitness - worst_fit) / (best_fit - worst_fit + c.EPSILON) + 1

        candidates = pop.positions.clone()
        current_mu_k = self.mu_k - (self.mu_k - 0.1) * (ctx.iteration / max(ctx.n_iterations, 1))
        for i in range(n):
            delta = torch.zeros_like(pop.positions[i])
            for j in range(n):
                if i == j or weights[i] >= weights[j]:
                    continue

                force = torch.maximum(weights[i] * self.mu_s, weights[j] * self.mu_s) - weights[i] * current_mu_k
                acceleration = force / _nonzero(weights[i] * pop.fitness.new_tensor(current_mu_k))
                acceleration = acceleration * (pop.positions[j] - pop.positions[i])
                noise = torch.randn_like(pop.positions[i])
                delta += 0.5 * acceleration * self.delta_t**2
                delta += self.alpha**t * self.beta * (pop.ub - pop.lb) * noise

            candidates[i] += delta

        for i in range(n):
            if torch.rand((), device=pop.device) < 0.5:
                candidates[i] = pop.best_position + torch.randn_like(candidates[i]) / t * (
                    pop.best_position - candidates[i]
                )

        candidates = candidates.clamp(min=pop.lb, max=pop.ub)
        candidate_fitness = fn(candidates)
        improved = candidate_fitness < pop.fitness
        pop.positions[improved] = candidates[improved]
        pop.fitness[improved] = candidate_fitness[improved]
        pop.update_best()
