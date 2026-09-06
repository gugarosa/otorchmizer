# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Evolutionary Programming.

References:
    X. Yao, Y. Liu, and G. Lin.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation (1999).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class EP(Optimizer):
    """Apply self-adaptive mutation and tournament selection with Evolutionary Programming."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> EP.")

        self.bout_size = 0.1
        self.clip_ratio = 0.05

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def bout_size(self) -> float:
        """Return the tournament bout ratio."""

        return self._bout_size

    @bout_size.setter
    def bout_size(self, bout_size: float) -> None:
        if not isinstance(bout_size, (float, int)):
            raise e.TypeError("`bout_size` must be a float or integer.")
        if not 0 <= bout_size <= 1:
            raise e.ValueError("`bout_size` must be between 0 and 1.")
        self._bout_size = bout_size

    @property
    def clip_ratio(self) -> float:
        """Return the strategy clipping ratio."""

        return self._clip_ratio

    @clip_ratio.setter
    def clip_ratio(self, clip_ratio: float) -> None:
        if not isinstance(clip_ratio, (float, int)):
            raise e.TypeError("`clip_ratio` must be a float or integer.")
        if not 0 <= clip_ratio <= 1:
            raise e.ValueError("`clip_ratio` must be between 0 and 1.")
        self._clip_ratio = clip_ratio

    def compile(self, population) -> None:
        """Initialize one mutation strategy per agent.

        Args:
            population: Population that defines state shape, bounds, and device.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        lb = population.lb.unsqueeze(0)
        ub = population.ub.unsqueeze(0)
        self.strategy = 0.05 * torch.rand(shape, device=population.device, dtype=population.dtype) * (ub - lb)

    def update(self, ctx: UpdateContext) -> None:
        """Create children and select the next generation by tournament wins.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        children = pop.positions + self.strategy * torch.randn_like(pop.positions)
        children = children.clamp(min=lb, max=ub)

        self.strategy = self.strategy + torch.randn_like(self.strategy) * torch.sqrt(self.strategy.abs() + 1e-10)
        self.strategy = self.strategy.clamp(min=lb, max=ub) * self.clip_ratio

        children_fitness = fn(children)

        all_pos = torch.cat([pop.positions, children], dim=0)
        all_fit = torch.cat([pop.fitness, children_fitness], dim=0)
        total = all_pos.shape[0]

        n_bouts = max(int(total * self.bout_size), 1)
        wins = torch.zeros(total, device=device, dtype=pop.dtype)

        for _ in range(n_bouts):
            opponents = torch.randint(0, total, (total,), device=device)
            wins += (all_fit < all_fit[opponents]).to(pop.dtype)

        _, selected = wins.topk(n, largest=True)
        pop.positions = all_pos[selected]
        pop.fitness = all_fit[selected]
        self.strategy = torch.cat([self.strategy, self.strategy], dim=0)[selected]
