# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Owl Search Algorithm.

References:
    M. Jain et al.
    Owl search algorithm: A novel nature-inspired heuristic paradigm.
    Soft Computing (2019).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class OSA(Optimizer):
    """Apply intensity- and distance-based movement toward the best owl."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> OSA.")

        self.beta = 1.9

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        """Return the exploration intensity."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        self._beta = beta

    def update(self, ctx: UpdateContext) -> None:
        """Move owls according to normalized intensity and prey distance.

        Args:
            ctx: Current optimization state.

        """

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
            intensity_change = intensity / (dist**2 + 1e-10) + torch.rand(1, device=device, dtype=pop.dtype)

            alpha = torch.rand(1, device=device, dtype=pop.dtype) * 0.5
            r = torch.rand(1, device=device, dtype=pop.dtype)

            if r.item() < 0.5:
                pop.positions[i] = pop.positions[i] + beta_t * intensity_change * torch.abs(
                    alpha * best_pos - pop.positions[i]
                )
            else:
                pop.positions[i] = pop.positions[i] - beta_t * intensity_change * torch.abs(
                    alpha * best_pos - pop.positions[i]
                )
