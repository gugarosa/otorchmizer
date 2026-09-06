# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Firefly Algorithm.

References:
    X.-S. Yang. Firefly algorithms for multimodal optimization.
    International Symposium on Stochastic Algorithms (2009).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FA(Optimizer):
    """Firefly Algorithm.

    Notes:
        Attraction uses a frozen population snapshot and sequential position overwrites.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> FA.")

        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Return the randomization coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        if alpha < 0:
            raise e.ValueError("`alpha` must be non-negative.")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the algorithm coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        if beta < 0:
            raise e.ValueError("`beta` must be non-negative.")
        self._beta = beta

    @property
    def gamma(self) -> float:
        """Return the gamma."""

        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not isinstance(gamma, (float, int)):
            raise e.TypeError("`gamma` must be a float or integer.")
        if gamma < 0:
            raise e.ValueError("`gamma` must be non-negative.")
        self._gamma = gamma

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        Notes:
            Brighter agents are processed in ascending fitness order.

        """

        pop = ctx.space.population
        n = pop.n_agents
        n_iterations = ctx.n_iterations

        delta = 1.0 - ((10e-4) / 0.9) ** (1.0 / n_iterations)
        self.alpha *= 1.0 - delta

        temp_positions = pop.positions.clone()
        temp_fitness = pop.fitness.clone()

        pos_flat = pop.positions.reshape(n, -1).clone()
        temp_flat = temp_positions.reshape(n, -1)

        sorted_idx = torch.argsort(temp_fitness)

        for j_idx in sorted_idx:
            j_fit = temp_fitness[j_idx]
            j_pos = temp_flat[j_idx]

            attracted = pop.fitness > j_fit

            if not attracted.any():
                continue

            diff = pos_flat[attracted] - j_pos.unsqueeze(0)
            dist = diff.norm(dim=1)

            beta_val = self.beta * torch.exp(-self.gamma * dist).unsqueeze(-1)

            r1 = torch.rand_like(pos_flat[attracted])
            pos_flat[attracted] = beta_val * (j_pos.unsqueeze(0) + pos_flat[attracted]) + self.alpha * (r1 - 0.5)

        pop.positions = pos_flat.reshape(pop.positions.shape)
