# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Jellyfish Search algorithms.

References:
    J.-S. Chou and D.-N. Truong.
    A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
    Applied Mathematics and Computation (2021).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class JS(Optimizer):
    """Jellyfish Search optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.eta = 4.0
        self.beta = 3.0
        self.gamma = 0.1

        super().__init__(params)

    @property
    def eta(self) -> float:
        """Return the logistic-map coefficient."""

        return self._eta

    @eta.setter
    def eta(self, eta: float) -> None:
        if not isinstance(eta, (float, int)):
            raise TypeError("`eta` must be a float or integer.")
        if not math.isfinite(eta) or not 0 < eta <= 4:
            raise ValueError("`eta` must be finite, greater than 0, and at most 4.")
        self._eta = eta

    @property
    def beta(self) -> float:
        """Return the ocean-current distribution coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("`beta` must be finite and positive.")
        self._beta = beta

    @property
    def gamma(self) -> float:
        """Return the passive-motion coefficient."""

        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not isinstance(gamma, (float, int)):
            raise TypeError("`gamma` must be a float or integer.")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("`gamma` must be finite and positive.")
        self._gamma = gamma

    def compile(self, population) -> None:
        """Initialize the population with a logistic chaotic map.

        Args:
            population: Population to initialize.

        """

        chaotic = torch.empty_like(population.positions)
        chaotic[0] = torch.rand_like(chaotic[0])
        for i in range(1, population.n_agents):
            chaotic[i] = self.eta * chaotic[i - 1] * (1 - chaotic[i - 1])
        chaotic = chaotic.clamp(0, 1)
        population.positions = population.lb.unsqueeze(0) + chaotic * (
            population.ub.unsqueeze(0) - population.lb.unsqueeze(0)
        )

    def _motion_a(self, pop) -> torch.Tensor:
        random = torch.rand(
            pop.n_agents,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        return self.gamma * random * (pop.ub.unsqueeze(0) - pop.lb.unsqueeze(0))

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        ratio = ctx.iteration / max(ctx.n_iterations, 1)
        control_random = torch.rand(n, device=pop.device, dtype=pop.dtype)
        control = torch.abs((1 - ratio) * (2 * control_random - 1))

        mean_position = pop.positions.mean(dim=0, keepdim=True)
        trend_random = torch.rand(n, 1, 1, device=pop.device, dtype=pop.dtype)
        trend = best - self.beta * trend_random * mean_position
        ocean_positions = (
            pop.positions
            + torch.rand(
                n,
                1,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            * trend
        )

        motion_selector = torch.rand(n, device=pop.device, dtype=pop.dtype)
        passive_positions = pop.positions + self._motion_a(pop)
        neighbours = torch.randint(0, n, (n,), device=pop.device)
        neighbour_positions = pop.positions[neighbours]
        toward_better = pop.fitness[neighbours] <= pop.fitness
        direction = torch.where(
            toward_better.view(n, 1, 1),
            neighbour_positions - pop.positions,
            pop.positions - neighbour_positions,
        )
        active_positions = (
            pop.positions
            + torch.rand(
                n,
                1,
                1,
                device=pop.device,
                dtype=pop.dtype,
            )
            * direction
        )
        swarm_positions = torch.where(
            (motion_selector > 1 - control).view(n, 1, 1),
            passive_positions,
            active_positions,
        )
        new_positions = torch.where(
            (control >= 0.5).view(n, 1, 1),
            ocean_positions,
            swarm_positions,
        )
        pop.positions = new_positions.clamp(min=pop.lb.unsqueeze(0), max=pop.ub.unsqueeze(0))


class NBJS(JS):
    """Jellyfish Search variant with bound-independent passive motion."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def _motion_a(self, pop) -> torch.Tensor:
        return self.gamma * torch.rand(
            pop.n_agents,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
