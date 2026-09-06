# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Butterfly Optimization Algorithm.

References:
    S. Arora and S. Singh.
    Butterfly optimization algorithm: a novel approach for global optimization.
    Soft Computing (2019).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BOA(Optimizer):
    """Butterfly Optimization Algorithm.

    Notes:
        Fragrance-based search with global and local phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

        super().__init__(params)

    @property
    def c(self) -> float:
        """Return the sensory modality."""

        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if not isinstance(c, (float, int)):
            raise TypeError("`c` must be a float or integer.")
        if not math.isfinite(c) or c < 0:
            raise ValueError("`c` must be finite and non-negative.")
        self._c = c

    @property
    def a(self) -> float:
        """Return the algorithm coefficient."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise TypeError("`a` must be a float or integer.")
        if not math.isfinite(a) or a < 0:
            raise ValueError("`a` must be finite and non-negative.")
        self._a = a

    @property
    def p(self) -> float:
        """Return the switch probability."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise TypeError("`p` must be a float or integer.")
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("`p` must be finite and between 0 and 1.")
        self._p = p

    def compile(self, population) -> None:
        """Initialize the per-agent fragrance buffer.

        Args:
            population: Population that determines buffer shape, device, and dtype.

        """

        self.fragrance = torch.zeros_like(population.fitness)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        self.fragrance = self.c * pop.fitness.abs() ** self.a
        fragrance = self.fragrance.view(n, 1, 1)
        random = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        global_pos = pop.positions + (random.square() * best - pop.positions) * fragrance

        first = torch.randint(0, n, (n,), device=device)
        if n > 1:
            second = torch.randint(0, n - 1, (n,), device=device)
            second += (second >= first).long()
        else:
            second = first
        local_pos = pop.positions + (random.square() * pop.positions[first] - pop.positions[second]) * fragrance
        use_global = (random.squeeze(-1).squeeze(-1) < self.p).view(n, 1, 1)
        pop.positions = torch.where(use_global, global_pos, local_pos)
