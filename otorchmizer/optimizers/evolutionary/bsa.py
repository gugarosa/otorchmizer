"""Backtracking Search Algorithm.

References:
    P. Civicioglu.
    Backtracking search optimization algorithm for numerical optimization problems.
    Applied Mathematics and Computation (2013).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BSA(Optimizer):
    """Backtracking Search Algorithm.

    Historical population-based mutation and crossover.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BSA.")

        self.F = 3.0
        self.mix_rate = 1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def F(self) -> float:
        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` should be a float or integer")
        self._F = F

    @property
    def mix_rate(self) -> int:
        return self._mix_rate

    @mix_rate.setter
    def mix_rate(self, mix_rate: int) -> None:
        if not isinstance(mix_rate, int):
            raise e.TypeError("`mix_rate` should be an integer")
        if mix_rate < 0:
            raise e.ValueError("`mix_rate` should be >= 0")
        self._mix_rate = mix_rate

    def compile(self, population) -> None:
        self.old_positions = population.positions.clone()

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Permute old population
        if torch.rand(1, device=device).item() < 0.5:
            self.old_positions = pop.positions.clone()

        perm = torch.randperm(n, device=device)
        old_shuffled = self.old_positions[perm]

        # Mutate: trial = pos + F * rand * (old - pos)
        r1 = torch.rand(1, device=device)
        trial = pop.positions + self.F * r1 * (old_shuffled - pop.positions)

        # Crossover
        cross_map = torch.ones(n, pop.n_variables, pop.n_dimensions, device=device, dtype=torch.bool)
        for i in range(n):
            # Keep mix_rate dimensions from original
            dims = torch.randperm(pop.n_variables, device=device)[:self.mix_rate]
            cross_map[i, dims, :] = False

        trial = torch.where(cross_map, trial, pop.positions)
        trial = trial.clamp(min=lb, max=ub)

        # Selection
        trial_fitness = fn(trial)
        improved = trial_fitness < pop.fitness
        pop.positions[improved] = trial[improved]
        pop.fitness[improved] = trial_fitness[improved]
