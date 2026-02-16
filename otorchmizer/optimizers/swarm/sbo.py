"""Satin Bowerbird Optimizer.

References:
    S. H. S. Moosavi and V. K. Bardsiri.
    Satin bowerbird optimizer: a new optimization algorithm to optimize
    ANFIS for software development effort estimation.
    Engineering Applications of Artificial Intelligence (2017).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SBO(Optimizer):
    """Satin Bowerbird Optimizer.

    Probability-based attraction and mutation.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SBO.")

        self.alpha = 0.94
        self.p_mutation = 0.05
        self.z = 0.02

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    @property
    def p_mutation(self) -> float:
        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        if not 0 <= p_mutation <= 1:
            raise e.ValueError("`p_mutation` should be between 0 and 1")
        self._p_mutation = p_mutation

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, z: float) -> None:
        if not isinstance(z, (float, int)):
            raise e.TypeError("`z` should be a float or integer")
        self._z = z

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Calculate probabilities based on fitness
        max_fit = pop.fitness.max()
        probs = (max_fit - pop.fitness + c.EPSILON)
        probs = probs / probs.sum()

        new_positions = pop.positions.clone()

        for i in range(n):
            # Select partner via roulette
            j = torch.multinomial(probs, 1).item()
            lam = self.alpha / (1 + probs[i])

            new_positions[i] = pop.positions[i] + lam * (
                (pop.positions[j] + pop.best_position) / 2 - pop.positions[i]
            )

        # Mutation
        mut_mask = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) < self.p_mutation
        sigma = self.z * (ub - lb)
        noise = torch.randn_like(new_positions) * sigma
        new_positions = torch.where(mut_mask, new_positions + noise, new_positions)

        new_positions = new_positions.clamp(min=lb, max=ub)
        new_fitness = fn(new_positions)

        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
