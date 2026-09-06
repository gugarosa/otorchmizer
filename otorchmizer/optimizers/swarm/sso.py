# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Simplified Swarm Optimization.

References:
    C. Bae et al.
    A new simplified swarm optimization using exchange local search scheme.
    International Journal of Innovative Computing, Information and Control (2012).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class SSO(Optimizer):
    """Simplified Swarm Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self._C_w = 0.1
        self._C_p = 0.4
        self._C_g = 0.9

        super().__init__(params)

    @property
    def C_w(self) -> float:
        """Return the probability threshold for retaining the current value."""

        return self._C_w

    @C_w.setter
    def C_w(self, C_w: float) -> None:
        self._validate_thresholds(C_w, self.C_p, self.C_g)
        self._C_w = C_w

    @property
    def C_p(self) -> float:
        """Return the cumulative personal-best probability threshold."""

        return self._C_p

    @C_p.setter
    def C_p(self, C_p: float) -> None:
        self._validate_thresholds(self.C_w, C_p, self.C_g)
        self._C_p = C_p

    @property
    def C_g(self) -> float:
        """Return the cumulative global-best probability threshold."""

        return self._C_g

    @C_g.setter
    def C_g(self, C_g: float) -> None:
        self._validate_thresholds(self.C_w, self.C_p, C_g)
        self._C_g = C_g

    @staticmethod
    def _validate_thresholds(C_w: float, C_p: float, C_g: float) -> None:
        for name, value in (("C_w", C_w), ("C_p", C_p), ("C_g", C_g)):
            if not isinstance(value, (float, int)):
                raise TypeError(f"`{name}` must be a float or integer.")
        if not 0 <= C_w <= 1:
            raise ValueError("`C_w` must be between 0 and 1.")
        if not C_w <= C_p <= 1:
            raise ValueError("`C_p` must be between `C_w` and 1.")
        if not C_p <= C_g <= 1:
            raise ValueError("`C_g` must be between `C_p` and 1.")

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply parameter overrides without transiently invalid coupled thresholds.

        Args:
            params: Attribute overrides applied to the optimizer.

        """

        supplied = dict(params or {})
        remaining = dict(supplied)
        C_w = remaining.pop("C_w", self.C_w)
        C_p = remaining.pop("C_p", self.C_p)
        C_g = remaining.pop("C_g", self.C_g)
        self._validate_thresholds(C_w, C_p, C_g)

        super().build(remaining)
        self._C_w, self._C_p, self._C_g = C_w, C_p, C_g
        self.params.update(
            {name: value for name, value in (("C_w", C_w), ("C_p", C_p), ("C_g", C_g)) if name in supplied}
        )

    def compile(self, population: Population) -> None:
        """Initialize persistent personal-best state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.local_position = torch.zeros_like(population.positions)
        self.local_fitness = torch.full_like(population.fitness, torch.inf)

    def evaluate(self, population: Population, function: Function) -> None:
        """Evaluate current positions and update personal and global bests.

        Args:
            population: Population to evaluate.
            function: Objective function applied to the population.

        """

        current_fitness = function(population.positions)
        improved = current_fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved]
        self.local_fitness[improved] = current_fitness[improved]
        population.fitness = current_fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        random = torch.rand(
            pop.n_agents,
            pop.n_variables,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        random_position = torch.rand_like(pop.positions) * (
            pop.ub.unsqueeze(0) - pop.lb.unsqueeze(0)
        ) + pop.lb.unsqueeze(0)
        personal = self.local_position
        global_best = pop.best_position.unsqueeze(0)
        pop.positions = torch.where(
            random < self.C_w,
            pop.positions,
            torch.where(
                random < self.C_p,
                personal,
                torch.where(random < self.C_g, global_best, random_position),
            ),
        )
