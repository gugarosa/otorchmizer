"""Differential Evolution.

References:
    R. Storn and K. Price.
    Differential Evolution – A Simple and Efficient Heuristic for
    Global Optimization over Continuous Spaces.
    Journal of Global Optimization (1997).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class DE(Optimizer):
    """Differential Evolution.

    Vectorized mutation, crossover, and selection.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> DE.")

        self.CR = 0.9
        self.F = 0.7

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def CR(self) -> float:
        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR` should be a float or integer")
        if not 0 <= CR <= 1:
            raise e.ValueError("`CR` should be between 0 and 1")
        self._CR = CR

    @property
    def F(self) -> float:
        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` should be a float or integer")
        if not 0 <= F <= 2:
            raise e.ValueError("`F` should be between 0 and 2")
        self._F = F

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Select 3 random distinct agents per individual
        idx_a = torch.randint(0, n, (n,), device=device)
        idx_b = torch.randint(0, n, (n,), device=device)
        idx_c = torch.randint(0, n, (n,), device=device)

        # Mutation: v = a + F * (b - c)
        mutant = pop.positions[idx_a] + self.F * (pop.positions[idx_b] - pop.positions[idx_c])

        # Crossover
        cr_mask = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) < self.CR
        # Ensure at least one dimension from mutant
        j_rand = torch.randint(0, pop.n_variables, (n,), device=device)
        for i in range(n):
            cr_mask[i, j_rand[i], :] = True

        trial = torch.where(cr_mask, mutant, pop.positions)
        trial = trial.clamp(min=lb, max=ub)

        # Selection
        trial_fitness = fn(trial)
        improved = trial_fitness < pop.fitness
        pop.positions[improved] = trial[improved]
        pop.fitness[improved] = trial_fitness[improved]
