# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Fish School Optimization.

References:
    C. J. A. Bastos Filho et al.
    A novel search algorithm based on fish school behavior.
    IEEE International Conference on Systems, Man and Cybernetics (2008).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FSO(Optimizer):
    """Fish School Optimization.

    Notes:
        Individual, feeding, instinctive, and volitive movement phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> FSO.")

        self.step_individual = 0.5
        self.step_volitive = 0.5
        self.min_weight = 1.0
        self.max_weight = 5000.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def step_individual(self) -> float:
        """Return the individual movement step."""

        return self._step_individual

    @step_individual.setter
    def step_individual(self, step_individual: float) -> None:
        if not isinstance(step_individual, (float, int)):
            raise e.TypeError("`step_individual` must be a float or integer.")
        self._step_individual = step_individual

    @property
    def step_volitive(self) -> float:
        """Return the volitive movement step."""

        return self._step_volitive

    @step_volitive.setter
    def step_volitive(self, step_volitive: float) -> None:
        if not isinstance(step_volitive, (float, int)):
            raise e.TypeError("`step_volitive` must be a float or integer.")
        self._step_volitive = step_volitive

    @property
    def min_weight(self) -> float:
        """Return the minimum fish weight."""

        return self._min_weight

    @min_weight.setter
    def min_weight(self, min_weight: float) -> None:
        if not isinstance(min_weight, (float, int)):
            raise e.TypeError("`min_weight` must be a float or integer.")
        self._min_weight = min_weight

    @property
    def max_weight(self) -> float:
        """Return the maximum fish weight."""

        return self._max_weight

    @max_weight.setter
    def max_weight(self, max_weight: float) -> None:
        if not isinstance(max_weight, (float, int)):
            raise e.TypeError("`max_weight` must be a float or integer.")
        self._max_weight = max_weight

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.weight = torch.full(
            (population.n_agents,), self.max_weight / 2.0, device=population.device, dtype=population.dtype
        )

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        old_fitness = pop.fitness.clone()
        total_weight_before = self.weight.sum()

        noise = (torch.rand_like(pop.positions) * 2 - 1) * self.step_individual
        new_positions = pop.positions + noise
        new_positions = new_positions.clamp(min=lb, max=ub)
        new_fitness = fn(new_positions)

        improved = new_fitness < pop.fitness
        delta_positions = torch.zeros_like(pop.positions)
        delta_positions[improved] = new_positions[improved] - pop.positions[improved]
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        delta_fitness = old_fitness - pop.fitness
        max_delta = delta_fitness.abs().max().clamp(min=1e-10)
        self.weight = self.weight + delta_fitness / max_delta
        self.weight = self.weight.clamp(min=self.min_weight, max=self.max_weight)

        delta_sum = (delta_fitness.view(n, 1, 1) * delta_positions).sum(dim=0)
        total_delta = delta_fitness.sum().clamp(min=1e-10)
        instinct = delta_sum / total_delta
        pop.positions = pop.positions + instinct.unsqueeze(0)
        pop.positions = pop.positions.clamp(min=lb, max=ub)

        total_weight_after = self.weight.sum()
        barycenter = (self.weight.view(n, 1, 1) * pop.positions).sum(dim=0) / total_weight_after

        direction = pop.positions - barycenter.unsqueeze(0)
        dist = torch.linalg.norm(direction.reshape(n, -1), dim=1).clamp(min=1e-10).view(n, 1, 1)
        step = (torch.rand_like(pop.positions) * 2 - 1) * self.step_volitive * direction / dist

        if total_weight_after > total_weight_before:
            pop.positions = pop.positions - step
        else:
            pop.positions = pop.positions + step

        pop.positions = pop.positions.clamp(min=lb, max=ub)
