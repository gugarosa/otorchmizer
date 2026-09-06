# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Electromagnetic Field Optimization.

References:
    H. Abedinpourshotorban et al.
    Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation (2016).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class EFO(Optimizer):
    """Electromagnetic Field Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the EFO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.positive_field = 0.1
        self.negative_field = 0.5
        self.ps_ratio = 0.1
        self.r_ratio = 0.4
        self.phi = (1 + 5**0.5) / 2
        self.RI = 0
        super().__init__(params)

    @property
    def positive_field(self) -> float:
        """Return the positive-field proportion.

        Returns:
            float: Current positive-field proportion.

        """

        return self._positive_field

    @positive_field.setter
    def positive_field(self, value: float) -> None:
        """Set the positive-field proportion.

        Args:
            value: New positive-field proportion.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`positive_field` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`positive_field` must be between 0 and 1.")
        self._positive_field = float(value)

    @property
    def negative_field(self) -> float:
        """Return the negative-field proportion.

        Returns:
            float: Current negative-field proportion.

        """

        return self._negative_field

    @negative_field.setter
    def negative_field(self, value: float) -> None:
        """Set the negative-field proportion.

        Args:
            value: New negative-field proportion.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`negative_field` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`negative_field` must be between 0 and 1.")
        self._negative_field = float(value)

    @property
    def ps_ratio(self) -> float:
        """Return the positive-selection probability.

        Returns:
            float: Current positive-selection probability.

        """

        return self._ps_ratio

    @ps_ratio.setter
    def ps_ratio(self, value: float) -> None:
        """Set the positive-selection probability.

        Args:
            value: New positive-selection probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`ps_ratio` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`ps_ratio` must be between 0 and 1.")
        self._ps_ratio = float(value)

    @property
    def r_ratio(self) -> float:
        """Return the random-reset probability.

        Returns:
            float: Current random-reset probability.

        """

        return self._r_ratio

    @r_ratio.setter
    def r_ratio(self, value: float) -> None:
        """Set the random-reset probability.

        Args:
            value: New random-reset probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`r_ratio` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`r_ratio` must be between 0 and 1.")
        self._r_ratio = float(value)

    @property
    def phi(self) -> float:
        """Return the golden-ratio coefficient.

        Returns:
            float: Current golden-ratio coefficient.

        """

        return self._phi

    @phi.setter
    def phi(self, value: float) -> None:
        """Set the golden-ratio coefficient.

        Args:
            value: New golden-ratio coefficient.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`phi` must be a float or integer.")
        self._phi = float(value)

    @property
    def RI(self) -> int:
        """Return the rotating reset index.

        Returns:
            int: Current reset index.

        """

        return self._RI

    @RI.setter
    def RI(self, value: int) -> None:
        """Set the rotating reset index.

        Args:
            value: New reset index.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is negative.

        """

        if not isinstance(value, int):
            raise TypeError("`RI` must be an integer.")
        if value < 0:
            raise ValueError("`RI` must be non-negative.")
        self._RI = value

    def compile(self, population) -> None:
        """Validate the population required by the electromagnetic fields.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            ValueError: If fewer than three agents are available.

        """

        if population.n_agents < 3:
            raise ValueError("`population.n_agents` must be at least 3 for EFO.")
        positive_end = max(int(population.n_agents * self.positive_field), 1)
        negative_start = min(
            max(int(population.n_agents * (1 - self.negative_field)), positive_end + 1),
            population.n_agents - 1,
        )
        if positive_end >= negative_start:
            raise ValueError("`positive_field` and `negative_field` must leave a non-empty neutral field.")
        if self.RI >= population.n_variables:
            raise ValueError("`RI` must be smaller than `population.n_variables`.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one EFO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        sorted_idx = torch.argsort(pop.fitness)
        positive_end = max(int(n * self.positive_field), 1)
        negative_start = min(max(int(n * (1 - self.negative_field)), positive_end + 1), n - 1)
        force = torch.rand((), device=device, dtype=pop.dtype)
        candidate = pop.positions[sorted_idx[0]].clone()

        for j in range(pop.n_variables):
            pos_rank = torch.randint(0, positive_end, (), device=device).item()
            neg_rank = torch.randint(negative_start, n, (), device=device).item()
            neutral_rank = torch.randint(positive_end, negative_start, (), device=device).item()
            pos = pop.positions[sorted_idx[pos_rank], j]
            neg = pop.positions[sorted_idx[neg_rank], j]
            neutral = pop.positions[sorted_idx[neutral_rank], j]

            if torch.rand((), device=device) < self.ps_ratio:
                candidate[j] = pos
            else:
                candidate[j] = neg + self.phi * force * (pos - neutral) - force * (neg - neutral)

        if torch.rand((), device=device) < self.r_ratio:
            candidate[self.RI] = (
                torch.rand_like(candidate[self.RI]) * (pop.ub[self.RI] - pop.lb[self.RI]) + pop.lb[self.RI]
            )
            self.RI = (self.RI + 1) % pop.n_variables

        candidate = candidate.clamp(min=pop.lb, max=pop.ub)
        candidate_fit = fn(candidate.unsqueeze(0))[0]
        worst_idx = sorted_idx[-1]
        if candidate_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = candidate
            pop.fitness[worst_idx] = candidate_fit
            pop.update_best()
