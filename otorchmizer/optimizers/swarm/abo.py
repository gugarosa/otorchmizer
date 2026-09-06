# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""African Buffalo Optimization.

References:
    J. Odili and M. Mohmad Kahar.
    Solving the Traveling Salesman's Problem Using the African Buffalo Optimization.
    Computational Intelligence and Neuroscience (2016).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ABO(Optimizer):
    """African Buffalo Optimization.

    Notes:
        Mimics the movement patterns of African buffaloes.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> ABO.")

        self.sunspot_ratio = 0.99
        self.starvation_ratio = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def sunspot_ratio(self) -> float:
        """Return the sunspot population proportion."""

        return self._sunspot_ratio

    @sunspot_ratio.setter
    def sunspot_ratio(self, sunspot_ratio: float) -> None:
        if not isinstance(sunspot_ratio, (float, int)):
            raise e.TypeError("`sunspot_ratio` must be a float or integer.")
        self._sunspot_ratio = sunspot_ratio

    @property
    def starvation_ratio(self) -> float:
        """Return the starvation reset probability."""

        return self._starvation_ratio

    @starvation_ratio.setter
    def starvation_ratio(self, starvation_ratio: float) -> None:
        if not isinstance(starvation_ratio, (float, int)):
            raise e.TypeError("`starvation_ratio` must be a float or integer.")
        self._starvation_ratio = starvation_ratio

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.w1 = torch.zeros(shape, device=population.device, dtype=population.dtype)
        self.w2 = torch.zeros(shape, device=population.device, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        lp1 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        lp2 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)

        self.w1 = self.w1 + lp1 * (best - pop.positions) + lp2 * (self.w1 - pop.positions)
        self.w2 = self.w2 / 2 + self.w1

        pop.positions = pop.positions + self.w2

        r = torch.rand(n, device=device, dtype=pop.dtype)
        starving = r < self.starvation_ratio
        if starving.any():
            lb = pop.lb.unsqueeze(0)
            ub = pop.ub.unsqueeze(0)
            n_s = starving.sum().item()
            pop.positions[starving] = (
                torch.rand(n_s, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * (ub - lb) + lb
            )
