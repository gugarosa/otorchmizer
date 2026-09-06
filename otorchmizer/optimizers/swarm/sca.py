# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sine Cosine Algorithm.

References:
    S. Mirjalili.
    SCA: a Sine Cosine Algorithm for solving optimization problems.
    Knowledge-Based Systems (2016).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SCA(Optimizer):
    """Sine Cosine Algorithm.

    Notes:
        Fully vectorized sine/cosine oscillation toward best position.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self._r_min = 0.0
        self._r_max = 2.0
        self.a = 3.0

        super().__init__(params)

    @property
    def r_min(self) -> float:
        """Return the minimum amplitude."""

        return self._r_min

    @r_min.setter
    def r_min(self, r_min: float) -> None:
        self._validate_range(r_min, self.r_max)
        self._r_min = r_min

    @property
    def r_max(self) -> float:
        """Return the maximum target-weight coefficient."""

        return self._r_max

    @r_max.setter
    def r_max(self, r_max: float) -> None:
        self._validate_range(self.r_min, r_max)
        self._r_max = r_max

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

    @staticmethod
    def _validate_range(r_min: float, r_max: float) -> None:
        for name, value in (("r_min", r_min), ("r_max", r_max)):
            if not isinstance(value, (float, int)):
                raise TypeError(f"`{name}` must be a float or integer.")
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
        if r_max < r_min:
            raise ValueError("`r_max` must be greater than or equal to `r_min`.")

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply parameter overrides without transiently invalid random-weight bounds.

        Args:
            params: Attribute overrides applied to the optimizer.

        """

        supplied = dict(params or {})
        remaining = dict(supplied)
        r_min = remaining.pop("r_min", self.r_min)
        r_max = remaining.pop("r_max", self.r_max)
        self._validate_range(r_min, r_max)

        super().build(remaining)
        self._r_min, self._r_max = r_min, r_max
        self.params.update({name: value for name, value in (("r_min", r_min), ("r_max", r_max)) if name in supplied})

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        best = pop.best_position.unsqueeze(0)

        r1 = self.a - ctx.iteration * self.a / max(ctx.n_iterations, 1)
        r2 = (
            torch.rand(
                pop.n_agents,
                pop.n_variables,
                pop.n_dimensions,
                device=device,
                dtype=pop.dtype,
            )
            * 2
            * torch.pi
        )
        r3 = self.r_min + torch.rand(
            pop.n_agents,
            pop.n_variables,
            pop.n_dimensions,
            device=device,
            dtype=pop.dtype,
        ) * (self.r_max - self.r_min)
        r4 = torch.rand(pop.n_agents, device=device, dtype=pop.dtype)

        sine_update = pop.positions + r1 * torch.sin(r2) * torch.abs(r3 * best - pop.positions)
        cosine_update = pop.positions + r1 * torch.cos(r2) * torch.abs(r3 * best - pop.positions)

        pop.positions = torch.where(
            (r4 < 0.5).view(pop.n_agents, 1, 1),
            sine_update,
            cosine_update,
        )
