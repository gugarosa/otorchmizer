"""Owl Search Algorithm.

References:
    M. Jain et al.
    Owl search algorithm: A novel nature-inspired heuristic paradigm.
    Soft Computing (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class OSA(Optimizer):
    """Owl Search Algorithm.

    Intensity and distance-based movement toward best.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> OSA.")

        self.beta = 1.9

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        self._beta = beta

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        sorted_idx = torch.argsort(pop.fitness)
        best_pos = pop.positions[sorted_idx[0]]
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]

        t = (ctx.iteration + 1) / max(ctx.n_iterations, 1)
        beta_t = self.beta - t * self.beta

        for i in range(n):
            intensity = (pop.fitness[i] - best_fit) / (worst_fit - best_fit + 1e-10)
            dist = g.euclidean_distance(pop.positions[i].reshape(-1), best_pos.reshape(-1))
            intensity_change = intensity / (dist ** 2 + 1e-10) + torch.randn(1, device=device) * 0.01

            alpha = torch.rand(1, device=device)
            r = torch.rand(1, device=device)

            if r.item() < 0.5:
                pop.positions[i] = pop.positions[i] + beta_t * intensity_change * torch.abs(alpha * best_pos - pop.positions[i])
            else:
                pop.positions[i] = pop.positions[i] - beta_t * intensity_change * torch.abs(alpha * best_pos - pop.positions[i])
