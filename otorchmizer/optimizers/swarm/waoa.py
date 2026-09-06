# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Weighted Aggregation Optimization Algorithm.

References:
    M. Barshandeh and F. Piri.
    WAOA: a meta-heuristic optimization algorithm based on
    the weighted aggregation optimization algorithm.
    Neural Computing and Applications (2019).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WAOA(Optimizer):
    """Weighted Aggregation Optimization Algorithm.

    Notes:
        Fitness-weighted leader-based search.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> WAOA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        pop.positions = pop.positions[sorted_idx]
        pop.fitness = pop.fitness[sorted_idx]

        worst_fit = pop.fitness[-1]
        best_fit = pop.fitness[0]
        w = (worst_fit - pop.fitness) / (worst_fit - best_fit + c.EPSILON)

        for i in range(n):
            r1 = torch.rand(1, 1, device=device, dtype=pop.dtype)
            r2 = torch.rand(1, 1, device=device, dtype=pop.dtype)

            j = torch.randint(0, n, (1,), device=device).item()
            k = torch.randint(0, n, (1,), device=device).item()

            if r1.item() < 0.5:
                pop.positions[i] = pop.positions[i] + r2 * (w[j] * pop.positions[j] - w[k] * pop.positions[k])
            else:
                pop.positions[i] = pop.positions[i] + r2 * (best.squeeze(0) - w[i] * pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
