# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Cuckoo Search.

References:
    X.-S. Yang and S. Deb.
    Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing (2009).

"""

from __future__ import annotations

import math
from typing import Any

import torch

import otorchmizer.math.distribution as d
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class CS(Optimizer):
    """Cuckoo Search optimizer.

    Notes:
        ``p`` is the probability of retaining a nest during differential replacement.
        Lévy displacement retains its independent Gaussian multiplier.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 1.0
        self.beta = 1.5
        self.p = 0.2

        super().__init__(params)

    @property
    def alpha(self) -> float:
        """Return the randomization coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("`alpha` must be finite and non-negative.")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the algorithm coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        if not math.isfinite(beta) or not 0 < beta <= 2:
            raise ValueError("`beta` must be finite, greater than 0, and at most 2.")
        self._beta = beta

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

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        levy = d.generate_levy_distribution(
            beta=self.beta,
            size=(n, pop.n_variables, 1),
            device=device,
            dtype=pop.dtype,
        )
        gaussian = torch.randn(
            n,
            pop.n_variables,
            1,
            device=device,
            dtype=pop.dtype,
        )
        step_size = self.alpha * levy * (pop.positions - best) * gaussian
        new_positions = pop.positions + step_size
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        replace_nest = torch.rand(n, device=device, dtype=pop.dtype) > self.p
        if replace_nest.any() and n > 1:
            first = torch.randint(0, n, (n,), device=device)
            second = torch.randint(0, n - 1, (n,), device=device)
            second += (second >= first).long()
            step = torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * (pop.positions[first] - pop.positions[second])
            new_pos2 = pop.positions + step
            new_pos2 = new_pos2.clamp(min=lb, max=ub)

            new_fit2 = fn(new_pos2)
            replace = replace_nest & (new_fit2 < pop.fitness)
            pop.positions[replace] = new_pos2[replace]
            pop.fitness[replace] = new_fit2[replace]
