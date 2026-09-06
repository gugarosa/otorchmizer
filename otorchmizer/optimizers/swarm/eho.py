# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Elephant Herding Optimization.

References:
    S. Deb, S. Fong, and Z. Tian.
    Elephant Herding Optimization.
    3rd International Symposium on Computational and Business Intelligence (2015).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class EHO(Optimizer):
    """Elephant Herding Optimization.

    Notes:
        Clan-based update with separation operator.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> EHO.")

        self.alpha = 0.5
        self.beta = 0.1
        self.n_clans = 2

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
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the algorithm coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        self._beta = beta

    @property
    def n_clans(self) -> int:
        """Return the number of clans."""

        return self._n_clans

    @n_clans.setter
    def n_clans(self, n_clans: int) -> None:
        if not isinstance(n_clans, int):
            raise e.TypeError("`n_clans` must be an integer.")
        if n_clans <= 0:
            raise e.ValueError("`n_clans` must be positive.")
        self._n_clans = n_clans

    def compile(self, population) -> None:
        """Validate population-dependent clan configuration.

        Args:
            population: Population whose agents will be divided into clans.

        Raises:
            ValueError: The number of clans exceeds the population size.

        """

        self._validate_population(population)

    def _validate_population(self, population) -> None:
        if self.n_clans > population.n_agents:
            raise e.ValueError("`n_clans` must not exceed `population.n_agents`.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        self._validate_population(pop)
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        n_per_clan = n // self.n_clans

        for clan_i in range(self.n_clans):
            start = clan_i * n_per_clan
            end = start + n_per_clan if clan_i < self.n_clans - 1 else n

            clan_pos = pop.positions[start:end]
            clan_fit = pop.fitness[start:end]
            clan_n = clan_pos.shape[0]

            best_idx = clan_fit.argmin()
            matriarch = clan_pos[best_idx].unsqueeze(0)

            r = torch.rand(clan_n, 1, 1, device=device, dtype=pop.dtype)
            new_clan = clan_pos + self.alpha * r * (matriarch - clan_pos)

            center = self.beta * clan_pos.mean(dim=0, keepdim=True)
            new_clan[best_idx] = center.squeeze(0)

            new_clan = new_clan.clamp(min=lb, max=ub)
            new_fit = fn(new_clan)

            improved = new_fit < clan_fit
            pop.positions[start:end][improved] = new_clan[improved]
            pop.fitness[start:end][improved] = new_fit[improved]

        worst_idx = pop.fitness.argmax()
        pop.positions[worst_idx] = torch.rand(pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * (
            ub.squeeze(0) - lb.squeeze(0)
        ) + lb.squeeze(0)
        pop.fitness[worst_idx] = fn(pop.positions[worst_idx].unsqueeze(0))[0]
