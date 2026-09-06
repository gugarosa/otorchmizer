# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Pigeon-Inspired Optimization.

References:
    H. Duan and P. Qiao.
    Pigeon-inspired optimization: A new swarm intelligence optimizer for air robot path planning.
    International Journal of Intelligent Computing and Cybernetics (2014).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class PIO(Optimizer):
    """Pigeon-Inspired Optimization.

    Notes:
        The landmark center retains the reference algorithm's extra active-pigeon divisor.
        An exact zero fitness sum uses ``active_mean / n_p``, matching the limit for equal positive weights.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self._n_c1 = 150
        self._n_c2 = 200
        self.R = 0.2

        super().__init__(params)

    @property
    def n_c1(self) -> int:
        """Return the final map-and-compass iteration."""

        return self._n_c1

    @n_c1.setter
    def n_c1(self, n_c1: int) -> None:
        self._validate_thresholds(n_c1, self.n_c2)
        self._n_c1 = n_c1

    @property
    def n_c2(self) -> int:
        """Return the final landmark iteration."""

        return self._n_c2

    @n_c2.setter
    def n_c2(self, n_c2: int) -> None:
        self._validate_thresholds(self.n_c1, n_c2)
        self._n_c2 = n_c2

    @staticmethod
    def _validate_thresholds(n_c1: int, n_c2: int) -> None:
        if not isinstance(n_c1, int):
            raise TypeError("`n_c1` must be an integer.")
        if not isinstance(n_c2, int):
            raise TypeError("`n_c2` must be an integer.")
        if n_c1 <= 0:
            raise ValueError("`n_c1` must be positive.")
        if n_c2 < n_c1:
            raise ValueError("`n_c2` must be greater than or equal to `n_c1`.")

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply parameter overrides without transiently invalid phase thresholds.

        Args:
            params: Attribute overrides applied to the optimizer.

        """

        supplied = dict(params or {})
        n_c1 = supplied.pop("n_c1", self.n_c1)
        n_c2 = supplied.pop("n_c2", self.n_c2)
        self._validate_thresholds(n_c1, n_c2)

        super().build(supplied)
        self._n_c1, self._n_c2 = n_c1, n_c2
        if params:
            self.params.update({name: value for name, value in (("n_c1", n_c1), ("n_c2", n_c2)) if name in params})

    @property
    def R(self) -> float:
        """Return the map-and-compass decay factor."""

        return self._R

    @R.setter
    def R(self, R: float) -> None:
        if not isinstance(R, (float, int)):
            raise TypeError("`R` must be a float or integer.")
        if R < 0:
            raise ValueError("`R` must be non-negative.")
        self._R = R

    @property
    def n_p(self) -> int:
        """Return the active pigeon count."""

        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, int):
            raise TypeError("`n_p` must be an integer.")
        if n_p <= 0:
            raise ValueError("`n_p` must be positive.")
        self._n_p = n_p

    def compile(self, population) -> None:
        """Initialize velocity and the active pigeon count.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.n_p = population.n_agents
        self.velocity = torch.zeros_like(population.positions)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        best = pop.best_position.unsqueeze(0)

        if ctx.iteration < self.n_c1:
            random = torch.rand(
                pop.n_agents,
                1,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            decay = pop.positions.new_tensor(-self.R * (ctx.iteration + 1)).exp()
            self.velocity = self.velocity * decay + random * (best - pop.positions)
            pop.positions = pop.positions + self.velocity
        elif ctx.iteration < self.n_c2:
            if not torch.isfinite(pop.fitness).all():
                raise ValueError("`population.fitness` must contain only finite values.")
            self.n_p = min(self.n_p // 2 + 1, pop.n_agents)
            order = torch.argsort(pop.fitness)
            active_positions = pop.positions[order[: self.n_p]]
            active_fitness = pop.fitness[order[: self.n_p]]
            scale = active_fitness.abs().max()
            if scale == 0:
                center = active_positions.mean(dim=0) / self.n_p
            else:
                normalized_fitness = active_fitness / scale
                fitness_sum = normalized_fitness.sum()
                if fitness_sum == 0:
                    center = active_positions.mean(dim=0) / self.n_p
                else:
                    center = (active_positions * normalized_fitness.view(-1, 1, 1)).sum(dim=0) / (
                        self.n_p * fitness_sum
                    )
            random = torch.rand(
                pop.n_agents,
                1,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            pop.positions = pop.positions + random * (center.unsqueeze(0) - pop.positions)
