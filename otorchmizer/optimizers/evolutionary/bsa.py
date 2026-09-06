# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Backtracking Search Algorithm.

References:
    P. Civicioglu.
    Backtracking search optimization algorithm for numerical optimization problems.
    Applied Mathematics and Computation (2013).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BSA(Optimizer):
    """Apply historical population mutation and crossover with Backtracking Search."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> BSA.")

        self.F = 3.0
        self.mix_rate = 1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def F(self) -> float:
        """Return the historical-population mutation scale."""

        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` must be a float or integer.")
        self._F = F

    @property
    def mix_rate(self) -> int:
        """Return the crossover-mask density control."""

        return self._mix_rate

    @mix_rate.setter
    def mix_rate(self, mix_rate: int) -> None:
        if not isinstance(mix_rate, int):
            raise e.TypeError("`mix_rate` must be an integer.")
        if mix_rate < 0:
            raise e.ValueError("`mix_rate` must be non-negative.")
        self._mix_rate = mix_rate

    def compile(self, population) -> None:
        """Initialize the historical population.

        Args:
            population: Population whose positions seed the historical state.

        """

        self.old_positions = population.positions.clone()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population through mutation, crossover, and selection.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        if torch.rand(1, device=device, dtype=pop.dtype).item() < 0.5:
            self.old_positions = pop.positions.clone()

        perm = torch.randperm(n, device=device)
        old_shuffled = self.old_positions[perm]

        r1 = torch.rand(1, device=device, dtype=pop.dtype)
        trial = pop.positions + self.F * r1 * (old_shuffled - pop.positions)

        cross_map = torch.ones(n, pop.n_variables, pop.n_dimensions, device=device, dtype=torch.bool)
        if torch.rand(1, device=device, dtype=pop.dtype).item() < torch.rand(1, device=device, dtype=pop.dtype).item():
            for i in range(n):
                non_crosses = int(
                    self.mix_rate * torch.rand(1, device=device, dtype=pop.dtype).item() * pop.n_variables
                )
                dims = torch.randperm(pop.n_variables, device=device)[:non_crosses]
                cross_map[i, dims, :] = False
        else:
            dims = torch.randint(0, pop.n_variables, (n,), device=device)
            cross_map[torch.arange(n, device=device), dims, :] = False

        trial = torch.where(cross_map, pop.positions, trial)
        trial = trial.clamp(min=lb, max=ub)

        trial_fitness = fn(trial)
        improved = trial_fitness < pop.fitness
        pop.positions[improved] = trial[improved]
        pop.fitness[improved] = trial_fitness[improved]
