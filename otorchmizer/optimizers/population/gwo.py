"""Grey Wolf Optimizer.

References:
    S. Mirjalili, S. M. Mirjalili, and A. Lewis.
    Grey Wolf Optimizer.
    Advances in Engineering Software (2014).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GWO(Optimizer):
    """Grey Wolf Optimizer.

    Fully vectorized alpha-beta-delta encircling and hunting.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> GWO.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        # Sort to find alpha, beta, delta
        sorted_idx = torch.argsort(pop.fitness)
        alpha = pop.positions[sorted_idx[0]].unsqueeze(0)
        beta = pop.positions[sorted_idx[1]].unsqueeze(0)
        delta = pop.positions[sorted_idx[2]].unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations - 1, 1)
        a = 2.0 - 2.0 * t  # linearly decreases from 2 to 0

        # Vectorized for all agents
        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = torch.abs(C1 * alpha - pop.positions)
        X1 = alpha - A1 * D_alpha

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = torch.abs(C2 * beta - pop.positions)
        X2 = beta - A2 * D_beta

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = torch.abs(C3 * delta - pop.positions)
        X3 = delta - A3 * D_delta

        pop.positions = (X1 + X2 + X3) / 3
