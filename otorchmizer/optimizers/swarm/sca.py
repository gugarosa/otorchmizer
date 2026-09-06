# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sine Cosine Algorithm.

References:
    S. Mirjalili.
    SCA: a Sine Cosine Algorithm for solving optimization problems.
    Knowledge-Based Systems (2016).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SCA(Optimizer):
    """Sine Cosine Algorithm.

    Notes:
        Fully vectorized sine/cosine oscillation toward best position.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> SCA.")

        self.r_min = 0.0
        self.a = 3.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def r_min(self) -> float:
        """Return the minimum amplitude."""

        return self._r_min

    @r_min.setter
    def r_min(self, r_min: float) -> None:
        if not isinstance(r_min, (float, int)):
            raise e.TypeError("`r_min` must be a float or integer.")
        self._r_min = r_min

    @property
    def a(self) -> float:
        """Return the algorithm coefficient."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` must be a float or integer.")
        self._a = a

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        r1 = self.a - t * (self.a - self.r_min)

        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * 2 * torch.pi
        r3 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * 2
        r4 = torch.rand(n, device=device, dtype=pop.dtype)

        use_sine = (r4 < 0.5).view(n, 1, 1)

        sine_update = pop.positions + r1 * torch.sin(r2) * torch.abs(r3 * best - pop.positions)
        cosine_update = pop.positions + r1 * torch.cos(r2) * torch.abs(r3 * best - pop.positions)

        pop.positions = torch.where(use_sine, sine_update, cosine_update)
