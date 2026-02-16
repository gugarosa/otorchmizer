"""Black Hole optimizer.

References:
    A. Hatamlou.
    Black hole: A new heuristic optimization approach for data clustering.
    Information Sciences (2013).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BH(Optimizer):
    """Black Hole optimizer.

    Stars attracted toward best solution with event horizon reset.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BH.")
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Move stars toward black hole
        r = torch.rand(n, 1, 1, device=device)
        pop.positions = pop.positions + r * (best - pop.positions)
        pop.positions = pop.positions.clamp(min=lb, max=ub)

        pop.fitness = fn(pop.positions)
        pop.update_best()

        # Event horizon
        cost = pop.best_fitness / (pop.fitness.sum() + 1e-10)
        radius = cost.item()

        flat = pop.positions.reshape(n, -1)
        best_flat = pop.best_position.reshape(1, -1)
        dist = torch.linalg.norm(flat - best_flat, dim=1)

        inside = dist < radius
        if inside.any():
            n_inside = inside.sum().item()
            pop.positions[inside] = torch.rand(n_inside, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb) + lb
            pop.fitness[inside] = fn(pop.positions[inside])
