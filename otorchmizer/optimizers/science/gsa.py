# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gravitational Search Algorithm.

References:
    E. Rashedi, H. Nezamabadi-pour, and S. Saryazdi.
    GSA: a gravitational search algorithm.
    Information Sciences (2009).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class GSA(Optimizer):
    """Gravitational Search Algorithm.

    Notes:
        Moves candidates using mass and force interactions under decaying gravity.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the GSA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.G = 2.467

        super().__init__(params)

    @property
    def G(self) -> float:
        """Return the initial gravity.

        Returns:
            float: Current initial gravity.

        """

        return self._G

    @G.setter
    def G(self, G: float) -> None:
        """Set the initial gravity.

        Args:
            G: New value for the initial gravity.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(G, (float, int)):
            raise TypeError("`G` must be a float or integer.")
        if G < 0:
            raise ValueError("`G` must be non-negative.")
        self._G = G

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = population.positions.new_zeros(shape)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one GSA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        t = ctx.iteration + 1
        G = self.G / t

        # Mass calculation
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        m = (pop.fitness - worst_fit) / (best_fit - worst_fit + c.EPSILON)
        M = m / (m.sum() + c.EPSILON)

        # Force calculation
        flat = pop.positions.reshape(n, -1)
        dist = torch.cdist(flat, flat).clamp(min=1e-10)

        force = torch.zeros_like(pop.positions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = torch.rand(1, device=device)
                    f = G * M[i] * M[j] / dist[i, j] * (pop.positions[j] - pop.positions[i])
                    force[i] += r * f

        # Acceleration
        accel = force / (M.view(n, 1, 1) + c.EPSILON)

        # Update velocity and position
        r = torch.rand(n, 1, 1, device=device)
        self.velocity = r * self.velocity + accel
        pop.positions = pop.positions + self.velocity
