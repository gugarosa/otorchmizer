# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sooty Tern Optimization Algorithm.

References:
    G. Dhiman and A. Kaur.
    STOA: A bio-inspired based optimization algorithm for industrial
    engineering problems.
    Engineering Applications of Artificial Intelligence (2019).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class STOA(Optimizer):
    """Sooty Tern Optimization Algorithm.

    Notes:
        Collision avoidance, convergence, and attack phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> STOA.")

        self.Cf = 2.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def Cf(self) -> float:
        """Return the collision-avoidance coefficient."""

        return self._Cf

    @Cf.setter
    def Cf(self, Cf: float) -> None:
        if not isinstance(Cf, (float, int)):
            raise e.TypeError("`Cf` must be a float or integer.")
        self._Cf = Cf

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

        Sa = self.Cf - t * self.Cf

        Cb = 0.5 * torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        diff = Sa * (best - torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * pop.positions)

        M = Cb * diff

        k = torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * 2 * torch.pi
        r_spiral = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)

        x = r_spiral * torch.sin(k)
        y = r_spiral * torch.cos(k)
        z = r_spiral * k

        pop.positions = M * (x + y + z) + best
