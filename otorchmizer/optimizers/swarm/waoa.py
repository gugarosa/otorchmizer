"""Weighted Aggregation Optimization Algorithm.

References:
    M. Barshandeh and F. Piri.
    WAOA: a meta-heuristic optimization algorithm based on
    the weighted aggregation optimization algorithm.
    Neural Computing and Applications (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WAOA(Optimizer):
    """Weighted Aggregation Optimization Algorithm.

    Fitness-weighted leader-based search.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> WAOA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Sorted fitness
        sorted_idx = torch.argsort(pop.fitness)
        pop.positions = pop.positions[sorted_idx]
        pop.fitness = pop.fitness[sorted_idx]

        # Fitness-based weight
        worst_fit = pop.fitness[-1]
        best_fit = pop.fitness[0]
        w = (worst_fit - pop.fitness) / (worst_fit - best_fit + c.EPSILON)

        for i in range(n):
            r1 = torch.rand(1, 1, device=device)
            r2 = torch.rand(1, 1, device=device)

            # Select two random agents
            j = torch.randint(0, n, (1,), device=device).item()
            k = torch.randint(0, n, (1,), device=device).item()

            if r1.item() < 0.5:
                pop.positions[i] = (
                    pop.positions[i]
                    + r2 * (w[j] * pop.positions[j] - w[k] * pop.positions[k])
                )
            else:
                pop.positions[i] = (
                    pop.positions[i]
                    + r2 * (best.squeeze(0) - w[i] * pop.positions[i])
                )

        pop.positions = pop.positions.clamp(min=lb, max=ub)
