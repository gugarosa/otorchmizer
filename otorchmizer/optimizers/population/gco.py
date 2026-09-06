# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Germinal Center Optimization.

References:
    C. Villaseñor et al.
    Germinal Center Optimization Algorithm.
    International Journal of Computational Intelligence Systems (2018).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GCO(Optimizer):
    """Apply dark-zone mutation and light-zone fitness-based life updates."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> GCO.")

        self.CR = 0.7
        self.F = 1.25

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def CR(self) -> float:
        """Return the mutation crossover probability."""

        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR` must be a float or integer.")
        if not 0 <= CR <= 1:
            raise e.ValueError("`CR` must be between 0 and 1.")
        self._CR = CR

    @property
    def F(self) -> float:
        """Return the mutation scale."""

        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` must be a float or integer.")
        self._F = F

    def compile(self, population) -> None:
        """Initialize cell life and selection counters.

        Args:
            population: Population that defines state length and device.

        """

        n = population.n_agents
        device = population.device
        self.life = torch.full((n,), 70.0, device=device, dtype=population.dtype)
        self.counter = torch.ones(n, device=device, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the dark-zone and light-zone phases.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for i in range(n):
            r = torch.rand(1, device=device, dtype=pop.dtype).item() * 100
            if r < self.life[i]:
                self.counter[i] += 1
            else:
                self.counter[i] = 1

            probs = self.counter / self.counter.sum()
            idx = torch.multinomial(probs, 3, replacement=True)

            new_pos = pop.positions[i].clone()
            for j in range(pop.n_variables):
                if torch.rand(1, device=device, dtype=pop.dtype).item() < self.CR:
                    new_pos[j] = pop.positions[idx[0], j] + self.F * (
                        pop.positions[idx[1], j] - pop.positions[idx[2], j]
                    )

            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit
                self.life[i] += 10

        min_fit = pop.fitness.min()
        max_fit = pop.fitness.max()
        self.life = 10 + 10 * (pop.fitness - max_fit) / (min_fit - max_fit + c.EPSILON)
