"""Invasive Weed Optimization.

References:
    A. R. Mehrabian and C. Lucas.
    A novel numerical optimization algorithm inspired from weed colonization.
    Ecological Informatics (2006).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class IWO(Optimizer):
    """Invasive Weed Optimization.

    Seed production, spatial dispersal, and competitive exclusion.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> IWO.")

        self.min_seeds = 0
        self.max_seeds = 5
        self.e = 2.0
        self.init_sigma = 3.0
        self.final_sigma = 0.001

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def min_seeds(self) -> int:
        return self._min_seeds

    @min_seeds.setter
    def min_seeds(self, min_seeds: int) -> None:
        if not isinstance(min_seeds, int):
            raise e.TypeError("`min_seeds` should be an integer")
        if min_seeds < 0:
            raise e.ValueError("`min_seeds` should be >= 0")
        self._min_seeds = min_seeds

    @property
    def max_seeds(self) -> int:
        return self._max_seeds

    @max_seeds.setter
    def max_seeds(self, max_seeds: int) -> None:
        if not isinstance(max_seeds, int):
            raise e.TypeError("`max_seeds` should be an integer")
        self._max_seeds = max_seeds

    @property
    def init_sigma(self) -> float:
        return self._init_sigma

    @init_sigma.setter
    def init_sigma(self, init_sigma: float) -> None:
        if not isinstance(init_sigma, (float, int)):
            raise e.TypeError("`init_sigma` should be a float or integer")
        if init_sigma < 0:
            raise e.ValueError("`init_sigma` should be >= 0")
        self._init_sigma = init_sigma

    @property
    def final_sigma(self) -> float:
        return self._final_sigma

    @final_sigma.setter
    def final_sigma(self, final_sigma: float) -> None:
        if not isinstance(final_sigma, (float, int)):
            raise e.TypeError("`final_sigma` should be a float or integer")
        if final_sigma < 0:
            raise e.ValueError("`final_sigma` should be >= 0")
        self._final_sigma = final_sigma

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        # Spatial dispersal
        coef = ((T - t) ** self.e) / (T ** self.e)
        sigma = coef * (self.init_sigma - self.final_sigma) + self.final_sigma

        # Sort by fitness
        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]

        offspring_list = []

        for i in range(n):
            ratio = (pop.fitness[i] - worst_fit) / (best_fit - worst_fit + 1e-10)
            n_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            if n_seeds > 0:
                parent = pop.positions[i].unsqueeze(0).expand(n_seeds, -1, -1)
                noise = torch.randn_like(parent) * sigma
                seeds = parent + noise
                seeds = seeds.clamp(min=lb, max=ub)
                offspring_list.append(seeds)

        if offspring_list:
            offspring = torch.cat(offspring_list, dim=0)
            offspring_fit = fn(offspring)

            # Combine and select best n
            all_pos = torch.cat([pop.positions, offspring], dim=0)
            all_fit = torch.cat([pop.fitness, offspring_fit], dim=0)
            best_idx = torch.argsort(all_fit)[:n]
            pop.positions = all_pos[best_idx]
            pop.fitness = all_fit[best_idx]
