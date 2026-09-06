# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Emperor Penguin Optimizer.

References:
    G. Dhiman and V. Kumar.
    Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
    Knowledge-Based Systems (2018).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class EPO(Optimizer):
    """Apply temperature-based emperor penguin huddle dynamics."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.f = 2.0
        self.l = 1.5

        super().__init__(params)

    @property
    def f(self) -> float:
        """Return the exploration control parameter."""

        return self._f

    @f.setter
    def f(self, f: float) -> None:
        if not isinstance(f, Real):
            raise TypeError("`f` must be a float or integer.")
        self._f = float(f)

    @property
    def l(self) -> float:  # noqa: E743
        """Return the exploitation control parameter."""

        return self._l

    @l.setter
    def l(self, l: float) -> None:  # noqa: E741, E743
        if not isinstance(l, Real):
            raise TypeError("`l` must be a float or integer.")
        self._l = float(l)

    def update(self, ctx: UpdateContext) -> None:
        """Move agents according to temperature and huddle geometry.

        Args:
            ctx: Current optimization state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        R = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        T_flag = (R < 0.5).to(pop.dtype)

        remaining = max(T - t, 1)
        T_p = T_flag + T / remaining

        P_grid = torch.abs(best - pop.positions)

        r1 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        C = torch.rand(
            n,
            pop.n_variables,
            1,
            device=device,
            dtype=pop.dtype,
        )

        A = 2 * (T_p + P_grid) * r1 - T_p

        S = (
            torch.abs(
                self.f * torch.exp(torch.tensor(-t / self.l, device=device, dtype=pop.dtype))
                - torch.exp(torch.tensor(-t, device=device, dtype=pop.dtype))
            )
        ) ** 2

        D_ep = torch.abs(S * best - C * pop.positions)

        pop.positions = best - A * D_ep
