# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Social Ski Driver.

References:
    A. Tharwat and T. Gabel.
    Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm.
    Neural Computing and Applications (2019).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SSD(Optimizer):
    """Social Ski Driver."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SSD optimizer.

        Args:
            params: Algorithm parameter overrides.

        Raises:
            TypeError: An exploration or decay coefficient is not numeric.
            ValueError: Exploration is negative or decay is outside [0, 1].

        """

        self.c = 2.0
        self.decay = 0.99
        super().__init__(params)
        if not isinstance(self.c, (float, int)) or not isinstance(self.decay, (float, int)):
            raise TypeError("`c` and `decay` must be floats or integers.")
        if self.c < 0 or not 0 <= self.decay <= 1:
            raise ValueError("`c` must be nonnegative and `decay` must be between 0 and 1.")

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        n = population.n_agents
        shape = (n, population.n_variables, population.n_dimensions)
        self.local_position = population.positions.new_zeros(shape)
        self.velocity = torch.rand_like(population.positions)
        self.local_fitness = population.fitness.new_full((n,), torch.inf)

    def evaluate(self, population, function) -> None:
        """Evaluate a population and update optimizer-specific best state.

        Args:
            population: Population whose tensors define the optimizer state.
            function: Objective function used to score the population.

        """

        fitness = function(population.positions)
        improved = fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved]
        self.local_fitness[improved] = fitness[improved]
        population.fitness = fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one SSD step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        alpha_pos = pop.positions[sorted_idx[0]]
        beta_pos = pop.positions[sorted_idx[1]] if n > 1 else alpha_pos
        gamma_pos = pop.positions[sorted_idx[2]] if n > 2 else beta_pos

        mean = (alpha_pos + beta_pos + gamma_pos) / 3

        for i in range(n):
            r1 = torch.rand(1, device=device, dtype=pop.dtype)
            r2 = torch.rand(1, device=device, dtype=pop.dtype)

            # Update position
            pop.positions[i] = pop.positions[i] + self.velocity[i]

            # Update velocity
            if r2.item() <= 0.5:
                self.velocity[i] = self.c * torch.sin(r1) * (self.local_position[i] - pop.positions[i]) + torch.sin(
                    r1
                ) * (mean - pop.positions[i])
            else:
                self.velocity[i] = self.c * torch.cos(r1) * (self.local_position[i] - pop.positions[i]) + torch.cos(
                    r1
                ) * (mean - pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        self.c *= self.decay
