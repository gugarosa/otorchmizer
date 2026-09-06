# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Aquila Optimizer.

References:
    L. Abualigah et al.
    Aquila Optimizer: A novel meta-heuristic optimization algorithm.
    Computers & Industrial Engineering (2021).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AO(Optimizer):
    """Apply four exploration and exploitation strategies inspired by Aquila hunting."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> AO.")

        self.alpha = 0.1
        self.delta = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Return the first exploitation adjustment coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        self._alpha = alpha

    @property
    def delta(self) -> float:
        """Return the second exploitation adjustment coefficient."""

        return self._delta

    @delta.setter
    def delta(self, delta: float) -> None:
        if not isinstance(delta, (float, int)):
            raise e.TypeError("`delta` must be a float or integer.")
        self._delta = delta

    def update(self, ctx: UpdateContext) -> None:
        """Move agents with the iteration-appropriate hunting strategies.

        Args:
            ctx: Current optimization state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        avg = pop.positions.mean(dim=0, keepdim=True)

        r1 = torch.rand(n, device=device, dtype=pop.dtype)
        r2 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)

        if t <= 2.0 / 3:
            use_strategy1 = r1 < 0.5
            s1 = use_strategy1.view(n, 1, 1)

            pos1 = best * (1 - t) + (avg - best * r2)

            levy = d.generate_levy_distribution(beta=1.5, size=pop.positions.shape, device=device).to(pop.dtype)
            j = torch.randint(0, n, (n,), device=device)
            pos2 = best * levy + pop.positions[j] + (torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * 2 - 1) * r2

            candidate = torch.where(s1, pos1, pos2)
        else:
            use_strategy3 = r1 < 0.5
            s3 = use_strategy3.view(n, 1, 1)

            pos3 = (best - avg) * self.alpha - r2 + ((ub - lb) * r2 + lb) * self.delta

            G1 = 2 * r2 - 1
            G2 = 2 * (1 - t)
            QF = t ** (G1 / ((1 - ctx.n_iterations) ** 2 + 1e-10))
            levy = d.generate_levy_distribution(beta=1.5, size=pop.positions.shape, device=device).to(pop.dtype)
            pos4 = QF * best - (G1 * pop.positions * r2) - G2 * levy + r2 * G1

            candidate = torch.where(s3, pos3, pos4)

        candidate = candidate.clamp(min=lb, max=ub)
        candidate_fitness = fn(candidate)
        improved = candidate_fitness < pop.fitness
        pop.positions[improved] = candidate[improved]
        pop.fitness[improved] = candidate_fitness[improved]
