"""Fruit Fly Optimization Algorithm.

References:
    W.-T. Pan.
    A new fruit fly optimization algorithm: taking the financial distress
    model as an example.
    Knowledge-Based Systems (2012).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FFOA(Optimizer):
    """Fruit Fly Optimization Algorithm.

    Osphresis (smell) and vision-based foraging.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> FFOA.")

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

        # Osphresis phase: random search around best
        noise = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        new_positions = best + noise

        # Smell concentration (inverse distance)
        dist = torch.sqrt(torch.sum(new_positions ** 2, dim=(1, 2)) + 1e-10)
        S = 1.0 / dist  # (n,)

        # Vision phase: evaluate smell-based positions
        smell_positions = new_positions.clone()
        for i in range(pop.n_variables):
            smell_positions[:, i, :] = S.view(n, 1) * new_positions[:, i, :]

        smell_positions = smell_positions.clamp(min=lb, max=ub)
        new_fitness = fn(smell_positions)

        # Update positions
        improved = new_fitness < pop.fitness
        pop.positions[improved] = smell_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
