# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Flying Squirrel Optimizer.

References:
    G. Azizyan et al.
    Flying Squirrel Optimizer: A novel SI-based optimization algorithm for engineering problems.
    Iranian Journal of Optimization (2019).

"""

from __future__ import annotations

import math
from typing import Any

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FSO(Optimizer):
    """Flying Squirrel Optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> FSO.")

        self.beta = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        """Return the initial Lévy-flight exponent."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        if not 0 < beta <= 2:
            raise e.ValueError("`beta` must be greater than 0 and at most 2.")
        self._beta = beta

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        mean_position = pop.positions.mean(dim=0, keepdim=True)
        srf = (-math.log1p(-1 / math.sqrt(ctx.iteration + 2))) ** 2
        progress = min((ctx.iteration + 1) / max(ctx.n_iterations, 1), 1)
        bef = self.beta + (2 - self.beta) * progress
        random_shape = (pop.n_agents, pop.n_variables, 1)
        random_step = torch.randn(random_shape, device=pop.device, dtype=pop.dtype) * srf + mean_position
        levy_step = d.generate_levy_distribution(
            beta=bef,
            size=random_shape,
            device=pop.device,
            dtype=pop.dtype,
        )
        best = pop.best_position.unsqueeze(0)
        candidates = pop.positions + random_step * levy_step * (pop.positions - best)
        candidates = candidates.clamp(min=pop.lb.unsqueeze(0), max=pop.ub.unsqueeze(0))
        candidate_fitness = fn(candidates)
        improved = candidate_fitness < pop.fitness
        pop.positions[improved] = candidates[improved]
        pop.fitness[improved] = candidate_fitness[improved]
        pop.update_best()
