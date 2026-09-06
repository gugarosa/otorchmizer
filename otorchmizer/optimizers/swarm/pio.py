# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Pigeon-Inspired Optimization.

References:
    H. Duan and P. Qiao.
    Pigeon-inspired optimization: a new swarm intelligence optimizer
    for air robot path planning.
    International Journal of Intelligent Computing and Cybernetics (2014).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PIO(Optimizer):
    """Pigeon-Inspired Optimization.

    Notes:
        Map & compass operator + landmark operator.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> PIO.")

        self.n_c = 0.0
        self.R = 0.2

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def n_c(self) -> float:
        """Return the phase-control coefficient."""

        return self._n_c

    @n_c.setter
    def n_c(self, n_c: float) -> None:
        if not isinstance(n_c, (float, int)):
            raise e.TypeError("`n_c` must be a float or integer.")
        self._n_c = n_c

    @property
    def R(self) -> float:
        """Return the map-and-compass factor."""

        return self._R

    @R.setter
    def R(self, R: float) -> None:
        if not isinstance(R, (float, int)):
            raise e.TypeError("`R` must be a float or integer.")
        self._R = R

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        if t < 0.5:
            r = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
            decay = pop.positions.new_tensor(-self.R * (ctx.iteration + 1))
            self.velocity = self.velocity * torch.exp(decay) + r * (best - pop.positions)
            pop.positions = pop.positions + self.velocity
        else:
            sorted_idx = torch.argsort(pop.fitness)
            n_active = max(int(n * (1 - t)), 2)

            top_positions = pop.positions[sorted_idx[:n_active]]
            top_fitness = pop.fitness[sorted_idx[:n_active]]

            weights = 1.0 / (top_fitness + 1e-10)
            weights = weights / weights.sum()
            center = (weights.view(-1, 1, 1) * top_positions).sum(dim=0, keepdim=True)

            r = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
            pop.positions = pop.positions + r * (center - pop.positions)
