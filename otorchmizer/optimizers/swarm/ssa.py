"""Salp Swarm Algorithm.

References:
    S. Mirjalili et al.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
    Advances in Engineering Software (2017).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SSA(Optimizer):
    """Salp Swarm Algorithm.

    Leader and follower phases fully vectorized.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SSA.")

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

        # Decay coefficient
        c1 = 2 * torch.exp(torch.tensor(-(4 * t) ** 2, device=device))

        # --- Leader update (first half) ---
        n_leaders = n // 2
        c2 = torch.rand(n_leaders, pop.n_variables, pop.n_dimensions, device=device)
        c3 = torch.rand(n_leaders, device=device)

        use_plus = (c3 >= 0.5).view(n_leaders, 1, 1)
        leader_pos = torch.where(use_plus, best + c1 * c2 * (ub - lb), best - c1 * c2 * (ub - lb))
        pop.positions[:n_leaders] = leader_pos

        # --- Follower update (second half) ---
        for i in range(n_leaders, n):
            pop.positions[i] = 0.5 * (pop.positions[i] + pop.positions[i - 1])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
