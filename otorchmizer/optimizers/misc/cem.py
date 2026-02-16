"""Cross-Entropy Method.

References:
    R. Y. Rubinstein.
    Optimization of computer simulation models with rare events.
    European Journal of Operational Research (1997).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class CEM(Optimizer):
    """Cross-Entropy Method.

    Distribution-based sampling with elite averaging.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> CEM.")

        self.n_updates = 5
        self.alpha = 0.7

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @n_updates.setter
    def n_updates(self, n_updates: int) -> None:
        if not isinstance(n_updates, int):
            raise e.TypeError("`n_updates` should be an integer")
        if n_updates <= 0:
            raise e.ValueError("`n_updates` should be > 0")
        self._n_updates = n_updates

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    def compile(self, population) -> None:
        lb = population.lb.squeeze(-1)
        ub = population.ub.squeeze(-1)
        device = population.device

        self.mean = torch.rand(population.n_variables, device=device) * (ub.squeeze(-1) - lb.squeeze(-1)) + lb.squeeze(-1)
        self.std = (ub.squeeze(-1) - lb.squeeze(-1)).clone()

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Sample new positions from current distribution
        for i in range(n):
            for j in range(pop.n_variables):
                pop.positions[i, j, :] = torch.randn(pop.n_dimensions, device=device) * self.std[j] + self.mean[j]

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        pop.fitness = fn(pop.positions)

        # Sort and select elite
        sorted_idx = torch.argsort(pop.fitness)
        n_elite = min(self.n_updates, n)
        elite = pop.positions[sorted_idx[:n_elite]]

        # Update mean and std with exponential moving average
        elite_mean = elite[:, :, 0].mean(dim=0)
        elite_std = ((elite[:, :, 0] - self.mean.unsqueeze(0)) ** 2).mean(dim=0).sqrt()

        self.mean = self.alpha * self.mean + (1 - self.alpha) * elite_mean
        self.std = self.alpha * self.std + (1 - self.alpha) * elite_std
