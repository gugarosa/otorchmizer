# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Fruit Fly Optimization Algorithm.

References:
    W.-T. Pan.
    A new fruit fly optimization algorithm: taking the financial distress
    model as an example.
    Knowledge-Based Systems (2012).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FFOA(Optimizer):
    """Fruit Fly Optimization Algorithm.

    Notes:
        Osphresis (smell) and vision-based foraging.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> FFOA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        noise = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype)
        new_positions = best + noise

        dist = torch.sqrt(torch.sum(new_positions**2, dim=(1, 2)) + 1e-10)
        S = 1.0 / dist

        smell_positions = new_positions.clone()
        for i in range(pop.n_variables):
            smell_positions[:, i, :] = S.view(n, 1) * new_positions[:, i, :]

        smell_positions = smell_positions.clamp(min=lb, max=ub)
        new_fitness = fn(smell_positions)

        improved = new_fitness < pop.fitness
        pop.positions[improved] = smell_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
