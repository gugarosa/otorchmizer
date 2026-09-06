# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Coyote Optimization Algorithm.

References:
    J. Pierezan and L. dos Santos Coelho.
    Coyote Optimization Algorithm: A new metaheuristic for global optimization problems.
    IEEE Congress on Evolutionary Computation (2018).
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class COA(Optimizer):
    """Apply pack-based cultural tendency with alpha leadership."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.n_p = 2

        super().__init__(params)

    @property
    def n_p(self) -> int:
        """Return the number of coyote packs."""

        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, Integral):
            raise TypeError("`n_p` must be an integer.")
        if n_p <= 0:
            raise ValueError("`n_p` must be positive.")
        self._n_p = int(n_p)

    def compile(self, population) -> None:
        """Calculate the nominal coyotes per pack.

        Args:
            population: Population whose agents are divided into packs.

        """

        if self.n_p > population.n_agents:
            raise ValueError("`n_p` must not exceed `population.n_agents`.")

        self.n_c = population.n_agents // self.n_p

    def update(self, ctx: UpdateContext) -> None:
        """Update each pack and optionally exchange coyotes.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for pack_i in range(self.n_p):
            start = pack_i * self.n_c
            end = start + self.n_c if pack_i < self.n_p - 1 else n

            pack_pos = pop.positions[start:end]
            pack_fit = pop.fitness[start:end]
            pack_n = pack_pos.shape[0]

            sorted_idx = torch.argsort(pack_fit)
            pack_pos = pack_pos[sorted_idx]
            pack_fit = pack_fit[sorted_idx]

            alpha = pack_pos[0].unsqueeze(0)

            tendency = pack_pos.median(dim=0).values.unsqueeze(0)

            for j in range(pack_n):
                cr1 = torch.randint(0, pack_n, (1,), device=device).item()
                cr2 = torch.randint(0, pack_n, (1,), device=device).item()

                r1 = torch.rand(1, 1, device=device, dtype=pop.dtype)
                r2 = torch.rand(1, 1, device=device, dtype=pop.dtype)

                lambda1 = alpha.squeeze(0) - pack_pos[cr1]
                lambda2 = tendency.squeeze(0) - pack_pos[cr2]

                new_pos = pack_pos[j] + r1 * lambda1 + r2 * lambda2
                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]

                if new_fit < pack_fit[j]:
                    pack_pos[j] = new_pos
                    pack_fit[j] = new_fit

            pop.positions[start:end] = pack_pos[torch.argsort(sorted_idx)]
            pop.fitness[start:end] = pack_fit[torch.argsort(sorted_idx)]

        p_e = 0.005 * n
        if torch.rand(1, device=device, dtype=pop.dtype).item() < p_e:
            i = torch.randint(0, n, (1,), device=device).item()
            j = torch.randint(0, n, (1,), device=device).item()
            pop.positions[[i, j]] = pop.positions[[j, i]]
            pop.fitness[[i, j]] = pop.fitness[[j, i]]
