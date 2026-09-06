# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Butterfly Optimization Algorithm.

References:
    S. Arora and S. Singh.
    Butterfly optimization algorithm: a novel approach for global optimization.
    Soft Computing (2019).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BOA(Optimizer):
    """Butterfly Optimization Algorithm.

    Notes:
        Fragrance-based search with global and local phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> BOA.")

        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def c(self) -> float:
        """Return the sensory modality."""

        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if not isinstance(c, (float, int)):
            raise e.TypeError("`c` must be a float or integer.")
        self._c = c

    @property
    def a(self) -> float:
        """Return the algorithm coefficient."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` must be a float or integer.")
        self._a = a

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
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        fragrance = self.c * pop.fitness.abs() ** self.a
        f = fragrance.view(n, 1, 1)

        r = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        prob = torch.rand(n, device=device, dtype=pop.dtype)

        global_pos = pop.positions + (r**2) * (best - pop.positions) * f

        j = torch.randint(0, n, (n,), device=device)
        k = torch.randint(0, n, (n,), device=device)
        local_pos = pop.positions + (r**2) * (pop.positions[j] - pop.positions[k]) * f

        use_global = (prob < self.p).view(n, 1, 1)
        pop.positions = torch.where(use_global, global_pos, local_pos)
