# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Invasive Weed Optimization.

References:
    A. R. Mehrabian and C. Lucas.
    A novel numerical optimization algorithm inspired from weed colonization.
    Ecological Informatics (2006).
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class IWO(Optimizer):
    """Apply seed production, spatial dispersal, and competitive exclusion."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.min_seeds = 0
        self.max_seeds = 5
        self.e = 2.0
        self.final_sigma = 0.001
        self.init_sigma = 3.0
        self.sigma = 0.0

        super().__init__(params)
        self._validate_parameters()

    def build(self, params: Mapping[str, Any] | None = None) -> None:
        """Apply parameter overrides after validating coupled values atomically.

        Args:
            params: Parameter overrides applied to the optimizer.

        """

        if params is not None and not isinstance(params, Mapping):
            raise TypeError("`params` must be a mapping.")
        if params is None:
            super().build(None)
            return

        min_seeds = params.get("min_seeds", self.min_seeds)
        max_seeds = params.get("max_seeds", self.max_seeds)
        final_sigma = params.get("final_sigma", self.final_sigma)
        init_sigma = params.get("init_sigma", self.init_sigma)
        if not isinstance(min_seeds, Integral):
            raise TypeError("`min_seeds` must be an integer.")
        if not isinstance(max_seeds, Integral):
            raise TypeError("`max_seeds` must be an integer.")
        if min_seeds < 0:
            raise ValueError("`min_seeds` must be non-negative.")
        if max_seeds < min_seeds:
            raise ValueError("`max_seeds` must be greater than or equal to `min_seeds`.")
        if not isinstance(final_sigma, Real):
            raise TypeError("`final_sigma` must be a float or integer.")
        if not isinstance(init_sigma, Real):
            raise TypeError("`init_sigma` must be a float or integer.")
        if final_sigma < 0:
            raise ValueError("`final_sigma` must be non-negative.")
        if init_sigma < final_sigma:
            raise ValueError("`init_sigma` must be greater than or equal to `final_sigma`.")

        coupled = {"min_seeds", "max_seeds", "final_sigma", "init_sigma"}
        super().build({key: value for key, value in params.items() if key not in coupled})
        self._min_seeds = int(min_seeds)
        self._max_seeds = int(max_seeds)
        self._final_sigma = float(final_sigma)
        self._init_sigma = float(init_sigma)
        self.params.update({key: value for key, value in params.items() if key in coupled})

    def _validate_parameters(self) -> None:
        if self.max_seeds < self.min_seeds:
            raise ValueError("`max_seeds` must be greater than or equal to `min_seeds`.")
        if self.init_sigma < self.final_sigma:
            raise ValueError("`init_sigma` must be greater than or equal to `final_sigma`.")

    @property
    def min_seeds(self) -> int:
        """Return the minimum seeds produced by an agent."""

        return self._min_seeds

    @min_seeds.setter
    def min_seeds(self, min_seeds: int) -> None:
        if not isinstance(min_seeds, Integral):
            raise TypeError("`min_seeds` must be an integer.")
        if min_seeds < 0:
            raise ValueError("`min_seeds` must be non-negative.")
        if hasattr(self, "_max_seeds") and min_seeds > self.max_seeds:
            raise ValueError("`min_seeds` must be less than or equal to `max_seeds`.")
        self._min_seeds = int(min_seeds)

    @property
    def max_seeds(self) -> int:
        """Return the maximum seeds produced by an agent."""

        return self._max_seeds

    @max_seeds.setter
    def max_seeds(self, max_seeds: int) -> None:
        if not isinstance(max_seeds, Integral):
            raise TypeError("`max_seeds` must be an integer.")
        if max_seeds < 0:
            raise ValueError("`max_seeds` must be non-negative.")
        if max_seeds < self.min_seeds:
            raise ValueError("`max_seeds` must be greater than or equal to `min_seeds`.")
        self._max_seeds = int(max_seeds)

    @property
    def e(self) -> float:
        """Return the spatial-dispersal decay exponent."""

        return self._e

    @e.setter
    def e(self, exponent: float) -> None:
        if not isinstance(exponent, Real):
            raise TypeError("`e` must be a float or integer.")
        if exponent < 0:
            raise ValueError("`e` must be non-negative.")
        self._e = float(exponent)

    @property
    def init_sigma(self) -> float:
        """Return the initial dispersal standard deviation."""

        return self._init_sigma

    @init_sigma.setter
    def init_sigma(self, init_sigma: float) -> None:
        if not isinstance(init_sigma, Real):
            raise TypeError("`init_sigma` must be a float or integer.")
        if init_sigma < 0:
            raise ValueError("`init_sigma` must be non-negative.")
        if init_sigma < self.final_sigma:
            raise ValueError("`init_sigma` must be greater than or equal to `final_sigma`.")
        self._init_sigma = float(init_sigma)

    @property
    def final_sigma(self) -> float:
        """Return the final dispersal standard deviation."""

        return self._final_sigma

    @final_sigma.setter
    def final_sigma(self, final_sigma: float) -> None:
        if not isinstance(final_sigma, Real):
            raise TypeError("`final_sigma` must be a float or integer.")
        if final_sigma < 0:
            raise ValueError("`final_sigma` must be non-negative.")
        if hasattr(self, "_init_sigma") and final_sigma > self.init_sigma:
            raise ValueError("`final_sigma` must be less than or equal to `init_sigma`.")
        self._final_sigma = float(final_sigma)

    @property
    def sigma(self) -> float:
        """Return the current spatial-dispersal scale."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if not isinstance(sigma, Real):
            raise TypeError("`sigma` must be a float or integer.")
        if sigma < 0:
            raise ValueError("`sigma` must be non-negative.")
        self._sigma = float(sigma)

    def update(self, ctx: UpdateContext) -> None:
        """Produce seeds and retain the best candidates.

        Args:
            ctx: Current optimization state and objective.

        """

        self._validate_parameters()

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        coef = ((T - t) ** self.e) / (T**self.e)
        self.sigma = coef * (self.init_sigma - self.final_sigma) + self.final_sigma

        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]

        offspring_list = []

        for i in range(n):
            ratio = (pop.fitness[i] - worst_fit) / (best_fit - worst_fit + 1e-10)
            n_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            if n_seeds > 0:
                parent = pop.positions[i].unsqueeze(0).expand(n_seeds, -1, -1)
                noise = torch.randn_like(parent) * self.sigma
                seeds = parent + noise
                seeds = seeds.clamp(min=lb, max=ub)
                offspring_list.append(seeds)

        if offspring_list:
            offspring = torch.cat(offspring_list, dim=0)
            offspring_fit = fn(offspring)

            all_pos = torch.cat([pop.positions, offspring], dim=0)
            all_fit = torch.cat([pop.fitness, offspring_fit], dim=0)
            best_idx = torch.argsort(all_fit)[:n]
            pop.positions = all_pos[best_idx]
            pop.fitness = all_fit[best_idx]
