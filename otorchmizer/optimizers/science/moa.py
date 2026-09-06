# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Magnetic Optimization Algorithm.

References:
    M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
    Swarm and Evolutionary Computation (2014).

"""

from __future__ import annotations

from math import isqrt
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class MOA(Optimizer):
    """Magnetic Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the MOA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 1.0
        self.rho = 2.0
        super().__init__(params)

    def compile(self, population) -> None:
        """Validate the square toroidal population.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            ValueError: If the population cannot form a square grid.

        """

        root = isqrt(population.n_agents)
        if root * root != population.n_agents:
            raise ValueError("`population.n_agents` must be a perfect square.")
        self.grid_size = root

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one MOA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        n = pop.n_agents
        root = self.grid_size

        if not torch.isfinite(pop.fitness).all():
            raise ValueError("`population.fitness` must contain finite values for MOA.")

        pop.sort_by_fitness()
        scale = pop.fitness.abs().max()
        scaled_fitness = torch.zeros_like(pop.fitness) if scale == 0 else pop.fitness / scale
        fitness_range = scaled_fitness.max() - scaled_fitness.min()
        norm_fit = (
            torch.zeros_like(scaled_fitness)
            if fitness_range == 0
            else (scaled_fitness - scaled_fitness.min()) / fitness_range
        )
        mass = self.alpha + self.rho * norm_fit
        positions = pop.positions.clone()
        velocities = torch.zeros_like(positions)
        for i in range(n):
            row, column = divmod(i, root)
            neighbors = (
                ((row - 1) % root) * root + column,
                ((row + 1) % root) * root + column,
                row * root + (column - 1) % root,
                row * root + (column + 1) % root,
            )
            contributions = []
            for j in neighbors:
                diff = positions[j] - positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                contributions.append(norm_fit[j] * diff / dist)

            force = torch.stack(contributions).sum(dim=0)
            velocities[i] = force / (mass[i] + 1e-10) * torch.rand((), device=pop.device, dtype=pop.dtype)

        pop.positions = positions + velocities
