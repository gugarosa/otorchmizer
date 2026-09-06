# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sooty Tern Optimization Algorithm.

References:
    G. Dhiman and A. Kaur.
    STOA: A bio-inspired based optimization algorithm for industrial engineering problems.
    Engineering Applications of Artificial Intelligence (2019).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class STOA(Optimizer):
    """Sooty Tern Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.Cf = 2.0
        self.u = 1.0
        self.v = 1.0

        super().__init__(params)

    @property
    def Cf(self) -> float:
        """Return the collision-avoidance coefficient."""

        return self._Cf

    @Cf.setter
    def Cf(self, Cf: float) -> None:
        self._Cf = self._validate_nonnegative("Cf", Cf)

    @property
    def u(self) -> float:
        """Return the spiral-radius scale."""

        return self._u

    @u.setter
    def u(self, u: float) -> None:
        self._u = self._validate_nonnegative("u", u)

    @property
    def v(self) -> float:
        """Return the spiral-growth coefficient."""

        return self._v

    @v.setter
    def v(self, v: float) -> None:
        self._v = self._validate_nonnegative("v", v)

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(f"`{name}` must be a float or integer.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"`{name}` must be finite and non-negative.")
        return value

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one migration and spiral-attack step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        progress = ctx.iteration / max(ctx.n_iterations, 1)
        avoidance = self.Cf * (1 - progress)
        collision = avoidance * pop.positions
        convergence = (
            0.5
            * torch.rand(
                pop.n_agents,
                1,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            * (pop.best_position.unsqueeze(0) - pop.positions)
        )
        distance = collision + convergence

        k = torch.rand(
            pop.n_agents,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        ) * (2 * torch.pi)
        radius = self.u * torch.exp(k * self.v)
        angle = torch.rand_like(k) * k
        spiral = radius * (torch.sin(angle) + torch.cos(angle) + angle)
        pop.positions = pop.best_position.unsqueeze(0) + distance * spiral
