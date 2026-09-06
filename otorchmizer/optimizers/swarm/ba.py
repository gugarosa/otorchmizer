# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Bat Algorithm.

References:
    X.-S. Yang. A new metaheuristic bat-inspired algorithm.
    Nature Inspired Cooperative Strategies for Optimization (2010).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BA(Optimizer):
    """Bat Algorithm.

    Notes:
        Frequency follows the published non-negative range. Candidate positions are committed only when the
        loudness test accepts an objective improvement, preserving matching position and fitness tensors.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self._f_min = 0.0
        self._f_max = 2.0
        self.A = 0.5
        self.r = 0.5

        super().__init__(params)

    @property
    def f_min(self) -> float:
        """Return the minimum frequency."""

        return self._f_min

    @f_min.setter
    def f_min(self, f_min: float) -> None:
        self._validate_frequencies(f_min, self.f_max)
        self._f_min = f_min

    @property
    def f_max(self) -> float:
        """Return the maximum frequency."""

        return self._f_max

    @f_max.setter
    def f_max(self, f_max: float) -> None:
        self._validate_frequencies(self.f_min, f_max)
        self._f_max = f_max

    @property
    def A(self) -> float:
        """Return the loudness parameter."""

        return self._A

    @A.setter
    def A(self, A: float) -> None:
        if not isinstance(A, (float, int)):
            raise TypeError("`A` must be a float or integer.")
        if not math.isfinite(A) or A < 0:
            raise ValueError("`A` must be finite and non-negative.")
        self._A = A

    @property
    def r(self) -> float:
        """Return the pulse rate."""

        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not isinstance(r, (float, int)):
            raise TypeError("`r` must be a float or integer.")
        if not math.isfinite(r) or r < 0:
            raise ValueError("`r` must be finite and non-negative.")
        self._r = r

    @staticmethod
    def _validate_frequencies(f_min: float, f_max: float) -> None:
        for name, value in (("f_min", f_min), ("f_max", f_max)):
            if not isinstance(value, (float, int)):
                raise TypeError(f"`{name}` must be a float or integer.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if f_max < f_min:
            raise ValueError("`f_max` must be greater than or equal to `f_min`.")

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply overrides without transiently invalid frequency bounds.

        Args:
            params: Attribute overrides applied to the optimizer.

        """

        supplied = dict(params or {})
        remaining = dict(supplied)
        f_min = remaining.pop("f_min", self.f_min)
        f_max = remaining.pop("f_max", self.f_max)
        self._validate_frequencies(f_min, f_max)

        super().build(remaining)
        self._f_min, self._f_max = f_min, f_max
        self.params.update({name: value for name, value in (("f_min", f_min), ("f_max", f_max)) if name in supplied})

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device, dtype=population.dtype)
        self.frequency = self.f_min + (self.f_max - self.f_min) * torch.rand(
            population.n_agents,
            device=population.device,
            dtype=population.dtype,
        )
        self.loudness = self.A * torch.rand(
            population.n_agents,
            device=population.device,
            dtype=population.dtype,
        )
        self.pulse_rate = self.r * torch.rand(
            population.n_agents,
            device=population.device,
            dtype=population.dtype,
        )

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

        beta = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        self.frequency = self.f_min + (self.f_max - self.f_min) * beta.squeeze()
        self.velocity = self.velocity + (pop.positions - best) * self.frequency.view(n, 1, 1)
        new_positions = pop.positions + self.velocity

        r_test = torch.rand(n, device=device, dtype=pop.dtype)
        local_mask = r_test > self.pulse_rate
        if local_mask.any():
            mean_loud = self.loudness.mean()
            noise = 0.001 * torch.randn_like(new_positions[local_mask]) * mean_loud
            new_positions[local_mask] = best + noise

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        r_accept = torch.rand(n, device=device, dtype=pop.dtype)
        accept = (new_fitness < pop.fitness) & (r_accept < self.loudness)

        pop.positions[accept] = new_positions[accept]
        pop.fitness[accept] = new_fitness[accept]

        self.loudness[accept] *= 0.9
        decay = pop.positions.new_tensor(-0.9 * (ctx.iteration + 1))
        self.pulse_rate[accept] = self.r * (1 - torch.exp(decay))
        pop.update_best()
