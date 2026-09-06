# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Artificial Butterfly Optimization.

References:
    X. Qi, Y. Zhu, and H. Zhang.
    A new meta-heuristic butterfly-inspired algorithm.
    Journal of Computational Science (2017).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ABO(Optimizer):
    """Artificial Butterfly Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> ABO.")

        self.sunspot_ratio = 0.9
        self.a = 2.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def sunspot_ratio(self) -> float:
        """Return the sunspot butterfly proportion."""

        return self._sunspot_ratio

    @sunspot_ratio.setter
    def sunspot_ratio(self, sunspot_ratio: float) -> None:
        if not isinstance(sunspot_ratio, (float, int)):
            raise e.TypeError("`sunspot_ratio` must be a float or integer.")
        if not 0 <= sunspot_ratio <= 1:
            raise e.ValueError("`sunspot_ratio` must be between 0 and 1.")
        self._sunspot_ratio = sunspot_ratio

    @property
    def a(self) -> float:
        """Return the free-flight coefficient."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` must be a float or integer.")
        if a < 0:
            raise e.ValueError("`a` must be non-negative.")
        self._a = a

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        lb = pop.lb
        ub = pop.ub

        order = torch.argsort(pop.fitness)
        pop.positions = pop.positions[order]
        pop.fitness = pop.fitness[order]
        n_sunspots = int(self.sunspot_ratio * n)

        for i in range(n):
            neighbour_limit = n if i < n_sunspots else n - n_sunspots
            neighbour = torch.randint(0, neighbour_limit, (), device=pop.device)
            variable = torch.randint(0, pop.n_variables, (), device=pop.device)
            flight = torch.rand((), device=pop.device, dtype=pop.dtype) * 2 - 1
            candidate = pop.positions[i].clone()
            candidate[variable] += (pop.positions[i, variable] - pop.positions[neighbour, variable]) * flight
            candidate = candidate.clamp(min=lb, max=ub)
            candidate_fitness = fn(candidate.unsqueeze(0))[0]

            if candidate_fitness < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fitness
            elif i >= n_sunspots:
                reference = torch.randint(0, n, (), device=pop.device)
                r1 = torch.rand((), device=pop.device, dtype=pop.dtype)
                distance = torch.abs(2 * r1 * pop.positions[reference] - pop.positions[i])
                r2 = torch.rand((), device=pop.device, dtype=pop.dtype)
                progress = min(ctx.iteration / max(ctx.n_iterations, 1), 1)
                decay = self.a * (1 - progress)
                free_flight = pop.positions[reference] - 2 * decay * r2 - decay * distance
                free_flight = free_flight.clamp(min=lb, max=ub)
                pop.positions[i] = free_flight
                pop.fitness[i] = fn(free_flight.unsqueeze(0))[0]

        pop.update_best()
