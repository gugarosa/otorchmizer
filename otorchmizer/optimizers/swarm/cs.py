# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Cuckoo Search.

References:
    X.-S. Yang and S. Deb.
    Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing (2009).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class CS(Optimizer):
    """Cuckoo Search optimizer.

    Notes:
        Vectorized Lévy flight exploration with fraction of worst nests abandoned.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> CS.")

        self.alpha = 1.0
        self.beta = 1.5
        self.p = 0.2

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Return the randomization coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the algorithm coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        self._beta = beta

    @property
    def p(self) -> float:
        """Return the switch probability."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` must be a float or integer.")
        if not 0 <= p <= 1:
            raise e.ValueError("`p` must be between 0 and 1.")
        self._p = p

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        levy = d.generate_levy_distribution(
            beta=self.beta,
            size=pop.positions.shape,
            device=device,
        ).to(dtype=pop.dtype)
        step_size = self.alpha * levy * (pop.positions - best)
        new_positions = pop.positions + step_size
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        abandon = torch.rand(n, device=device, dtype=pop.dtype) < self.p
        if abandon.any():
            perm1 = torch.randperm(n, device=device)
            perm2 = torch.randperm(n, device=device)
            step = torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * (pop.positions[perm1] - pop.positions[perm2])
            new_pos2 = pop.positions + step
            new_pos2 = new_pos2.clamp(min=lb, max=ub)

            new_fit2 = fn(new_pos2)
            replace = abandon & (new_fit2 < pop.fitness)
            pop.positions[replace] = new_pos2[replace]
            pop.fitness[replace] = new_fit2[replace]
