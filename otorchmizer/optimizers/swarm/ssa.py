# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Salp Swarm Algorithm.

References:
    S. Mirjalili et al.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
    Advances in Engineering Software (2017).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SSA(Optimizer):
    """Salp Swarm Algorithm.

    Notes:
        Uses leader and follower phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        c1 = pop.positions.new_tensor(2 * math.exp(-((4 * t) ** 2)))
        c2 = torch.rand_like(pop.positions[:1])
        c3 = torch.rand_like(pop.positions[:1])
        displacement = c1 * ((ub - lb) * c2 + lb)
        pop.positions[:1] = torch.where(c3 < 0.5, best + displacement, best - displacement)

        for i in range(1, n):
            pop.positions[i] = 0.5 * (pop.positions[i] + pop.positions[i - 1])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
