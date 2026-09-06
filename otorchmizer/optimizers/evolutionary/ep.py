# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Evolutionary Programming.

References:
    X. Yao, Y. Liu, and G. Lin.
    Evolutionary programming made faster.
    IEEE Transactions on Evolutionary Computation (1999).

Notes:
    This implements Gaussian classical evolutionary programming (CEP): offspring positions use the parent's
    unadapted strategy, while the separately adapted log-normal strategy is inherited after selection. Fast
    evolutionary programming (FEP) instead uses Cauchy-distributed position mutations and is not implemented here.
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

        n_parameters = pop.n_variables * pop.n_dimensions
        tau_global = 1 / (2 * n_parameters) ** 0.5
        tau_local = 1 / (2 * n_parameters**0.5) ** 0.5
        global_noise = torch.randn(n, 1, 1, device=device, dtype=pop.dtype)
        local_noise = torch.randn_like(self.strategy)
        child_strategy = self.strategy * torch.exp(tau_global * global_noise + tau_local * local_noise)
        strategy_limit = (ub - lb) * self.clip_ratio
        child_strategy = torch.minimum(child_strategy, strategy_limit)

        children = pop.positions + self.strategy * torch.randn_like(pop.positions)
        children = children.clamp(min=lb, max=ub)
        children_fitness = fn(children)

        all_pos = torch.cat([pop.positions, children], dim=0)
        all_fit = torch.cat([pop.fitness, children_fitness], dim=0)
        all_strategy = torch.cat([self.strategy, child_strategy], dim=0)
        total = all_pos.shape[0]
        best_idx = all_fit.argmin()
        if all_fit[best_idx] < pop.best_fitness:
            pop.best_position = all_pos[best_idx].clone()
            pop.best_fitness = all_fit[best_idx].clone()

        n_bouts = max(int(n * self.bout_size), 1)
        wins = torch.zeros(total, device=device, dtype=pop.dtype)

        for _ in range(n_bouts):
            opponents = torch.randint(0, total, (total,), device=device)
            wins += (all_fit < all_fit[opponents]).to(pop.dtype)

        _, selected = wins.topk(n, largest=True)
        pop.positions = all_pos[selected]
        pop.fitness = all_fit[selected]
        self.strategy = all_strategy[selected]
