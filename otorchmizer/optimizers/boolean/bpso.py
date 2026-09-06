# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Boolean Particle Swarm Optimization.

Compilation allocates Boolean velocity and personal-best arrays for the population.
Evaluation records improvements to personal and global best positions.

References:
    F. Afshinmanesh, A. Marandi and A. Rahimi-Kian.
    A Novel Binary Particle Swarm Optimization Method Using Artificial Immune System.
    IEEE International Conference on Smart Technologies (2005).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BPSO(Optimizer):
    """Boolean Particle Swarm Optimization.

    Notes:
        Uses XOR-based velocity and position updates in the Boolean domain.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the BPSO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c1 = 1.0
        self.c2 = 1.0
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        n = population.n_agents
        shape = (n, population.n_variables, population.n_dimensions)
        device = population.device
        self.local_position = torch.zeros(shape, dtype=torch.bool, device=device)
        self.velocity = torch.zeros(shape, dtype=torch.bool, device=device)
        self.local_fitness = population.fitness.new_full((n,), torch.inf)

    def evaluate(self, population, function) -> None:
        """Evaluate a population and update optimizer-specific best state.

        Args:
            population: Population whose tensors define the optimizer state.
            function: Objective function used to score the population.

        """

        fitness = function(population.positions)
        improved = fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved].bool()
        self.local_fitness[improved] = fitness[improved]
        population.fitness = fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one BPSO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.bool()

        c1_b = torch.tensor(self.c1, device=device).bool() if self.c1 else torch.zeros(1, device=device).bool()
        c2_b = torch.tensor(self.c2, device=device).bool() if self.c2 else torch.zeros(1, device=device).bool()

        for i in range(n):
            pos = pop.positions[i].bool()
            r1 = torch.round(torch.rand_like(pop.positions[i])).bool()
            r2 = torch.round(torch.rand_like(pop.positions[i])).bool()

            local_partial = c1_b & (r1 ^ (self.local_position[i] ^ pos))
            global_partial = c2_b & (r2 ^ (best ^ pos))

            self.velocity[i] = local_partial | global_partial
            pop.positions[i] = (pos ^ self.velocity[i]).to(dtype=pop.dtype)
