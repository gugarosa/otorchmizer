# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Artificial Ecosystem-based Optimization.

References:
    W. Zhao, L. Wang, and Z. Zhang.
    Artificial ecosystem-based optimization: a novel nature-inspired
    meta-heuristic algorithm.
    Neural Computing and Applications (2020).
"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AEO(Optimizer):
    """Apply ecosystem production, consumption, and decomposition phases."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> AEO.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population through composition and decomposition.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Descending fitness assigns the producer and consumer roles
        sorted_idx = torch.argsort(pop.fitness, descending=True)
        positions_sorted = pop.positions[sorted_idx].clone()
        fitness_sorted = pop.fitness[sorted_idx].clone()

        # Composition phase
        for i in range(n):
            new_pos = positions_sorted[i].clone()
            r1 = torch.rand(1, device=device, dtype=pop.dtype).item()

            if i == 0:
                alpha = (1 - t) * torch.rand(1, device=device, dtype=pop.dtype)
                new_pos = (
                    (1 - alpha) * best.squeeze(0)
                    + alpha * torch.rand_like(new_pos) * (ub.squeeze(0) - lb.squeeze(0))
                    + lb.squeeze(0)
                )
            elif r1 < 1.0 / 3:
                C = (
                    0.5
                    * torch.randn(1, device=device, dtype=pop.dtype)
                    / torch.abs(torch.randn(1, device=device, dtype=pop.dtype) + 1e-10)
                )
                new_pos = positions_sorted[i] + C * (positions_sorted[i] - positions_sorted[0])
            elif r1 < 2.0 / 3:
                C = (
                    0.5
                    * torch.randn(1, device=device, dtype=pop.dtype)
                    / torch.abs(torch.randn(1, device=device, dtype=pop.dtype) + 1e-10)
                )
                j = torch.randint(1, n, (1,), device=device).item()
                r2 = torch.rand(1, device=device, dtype=pop.dtype)
                new_pos = positions_sorted[i] + C * (
                    r2 * (positions_sorted[i] - positions_sorted[0])
                    + (1 - r2) * (positions_sorted[i] - positions_sorted[j])
                )
            else:
                C = (
                    0.5
                    * torch.randn(1, device=device, dtype=pop.dtype)
                    / torch.abs(torch.randn(1, device=device, dtype=pop.dtype) + 1e-10)
                )
                j = torch.randint(1, n, (1,), device=device).item()
                new_pos = positions_sorted[i] + C * (positions_sorted[i] - positions_sorted[j])

            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < fitness_sorted[i]:
                positions_sorted[i] = new_pos
                fitness_sorted[i] = new_fit

        # Decomposition phase
        r3 = torch.rand(n, device=device, dtype=pop.dtype)
        D = 3 * torch.randn(n, 1, 1, device=device, dtype=pop.dtype)
        e = r3.view(n, 1, 1) * torch.randint(1, 3, (n, 1, 1), device=device).to(pop.dtype) - 1
        h = 2 * r3.view(n, 1, 1) - 1

        decomp_pos = best + D * (e * best - h * positions_sorted)
        decomp_pos = decomp_pos.clamp(min=lb, max=ub)
        decomp_fit = fn(decomp_pos)

        improved = decomp_fit < fitness_sorted
        positions_sorted[improved] = decomp_pos[improved]
        fitness_sorted[improved] = decomp_fit[improved]

        # Restore the original agent order
        pop.positions[sorted_idx] = positions_sorted
        pop.fitness[sorted_idx] = fitness_sorted
