# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Grey Wolf Optimizer.

References:
    S. Mirjalili, S. M. Mirjalili, and A. Lewis.
    Grey Wolf Optimizer.
    Advances in Engineering Software (2014).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GWO(Optimizer):
    """Apply vectorized alpha-beta-delta encircling and hunting."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> GWO.")

        super().__init__(params)

        logger.info("Class overrided.")

    def compile(self, population) -> None:
        """Validate that alpha, beta, and delta leaders are available.

        Args:
            population: Population whose size is validated.

        Raises:
            ValueError: The population contains fewer than three agents.

        """

        if population.n_agents < 3:
            raise e.ValueError("`population.n_agents` must be at least 3.")

    def update(self, ctx: UpdateContext) -> None:
        """Move all wolves relative to the three leading candidates.

        Args:
            ctx: Current optimization state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        sorted_idx = torch.argsort(pop.fitness)
        alpha = pop.positions[sorted_idx[0]].unsqueeze(0)
        beta = pop.positions[sorted_idx[1]].unsqueeze(0)
        delta = pop.positions[sorted_idx[2]].unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations - 1, 1)
        a = 2.0 - 2.0 * t

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = torch.abs(C1 * alpha - pop.positions)
        X1 = alpha - A1 * D_alpha

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = torch.abs(C2 * beta - pop.positions)
        X2 = beta - A2 * D_beta

        r1 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        r2 = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = torch.abs(C3 * delta - pop.positions)
        X3 = delta - A3 * D_delta

        pop.positions = (X1 + X2 + X3) / 3
