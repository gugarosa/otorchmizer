# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Atom Search Optimization.

References:
    W. Zhao, L. Wang, and Z. Zhang.
    Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem.
    Knowledge-Based Systems (2019).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ASO(Optimizer):
    """Atom Search Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ASO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> ASO.")
        self.alpha = 50.0
        self.beta = 0.2
        super().__init__(params)
        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Return the alpha coefficient.

        Returns:
            float: Current alpha coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set the alpha coefficient.

        Args:
            alpha: New value for the alpha coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the beta coefficient.

        Returns:
            float: Current beta coefficient.

        """

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        """Set the beta coefficient.

        Args:
            beta: New value for the beta coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        self._beta = beta

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = population.positions.new_zeros(shape)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ASO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        G = torch.exp(torch.tensor(-20.0 * t, device=device))

        # Mass
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        m = torch.exp(-(pop.fitness - best_fit) / (worst_fit - best_fit + c.EPSILON))
        M = m / (m.sum() + c.EPSILON)

        # K best
        K = max(int(n * (1 - t)), 2)
        sorted_idx = torch.argsort(pop.fitness)[:K]

        # Acceleration
        accel = torch.zeros_like(pop.positions)
        for i in range(n):
            for j_idx in sorted_idx:
                if j_idx == i:
                    continue
                diff = pop.positions[j_idx] - pop.positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                r = torch.rand(1, device=device)
                accel[i] += r * G * M[j_idx] * diff / dist

        r = torch.rand(n, 1, 1, device=device)
        self.velocity = r * self.velocity + accel + self.beta * (best - pop.positions)
        pop.positions = pop.positions + self.velocity
