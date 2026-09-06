# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Grasshopper Optimization Algorithm.

References:
    S. Saremi, S. Mirjalili, and A. Lewis.
    Grasshopper Optimisation Algorithm: Theory and application.
    Advances in Engineering Software (2017).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class GOA(Optimizer):
    """Grasshopper Optimization Algorithm.

    Notes:
        Social interaction forces with pairwise distance computation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c_min = 0.00001
        self.c_max = 1.0
        self.f = 0.5
        self.l = 1.5

        super().__init__(params)

    @property
    def c_min(self) -> float:
        """Return the minimum comfort coefficient."""

        return self._c_min

    @c_min.setter
    def c_min(self, c_min: float) -> None:
        if not isinstance(c_min, (float, int)):
            raise TypeError("`c_min` must be a float or integer.")
        self._c_min = c_min

    @property
    def c_max(self) -> float:
        """Return the maximum comfort coefficient."""

        return self._c_max

    @c_max.setter
    def c_max(self, c_max: float) -> None:
        if not isinstance(c_max, (float, int)):
            raise TypeError("`c_max` must be a float or integer.")
        self._c_max = c_max

    @property
    def f(self) -> float:
        """Return the attraction intensity."""

        return self._f

    @f.setter
    def f(self, f: float) -> None:
        if not isinstance(f, (float, int)):
            raise TypeError("`f` must be a float or integer.")
        self._f = f

    @property
    def l(self) -> float:  # noqa: E743
        """Return the attraction length scale."""

        return self._l

    @l.setter
    def l(self, l: float) -> None:  # noqa: E741, E743
        if not isinstance(l, (float, int)):
            raise TypeError("`l` must be a float or integer.")
        self._l = l

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        c = self.c_max - t * (self.c_max - self.c_min)

        flat = pop.positions.reshape(n, -1)
        dist_matrix = torch.cdist(flat, flat).clamp(min=torch.finfo(pop.dtype).tiny)
        social_distance = 2 + torch.fmod(dist_matrix, 2)

        s = self.f * torch.exp(-social_distance / self.l) - torch.exp(-social_distance)
        s.fill_diagonal_(0)

        diff = pop.positions.unsqueeze(0) - pop.positions.unsqueeze(1)
        norm = dist_matrix.unsqueeze(-1).unsqueeze(-1)
        direction = diff / norm

        force = (c * (ub - lb) / 2 * s.unsqueeze(-1).unsqueeze(-1) * direction).sum(dim=1)
        pop.positions = c * force + best
        pop.positions = pop.positions.clamp(min=lb, max=ub)
