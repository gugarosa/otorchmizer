# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Boolean-based optimizers: BMRFO, BPSO, UMDA."""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BMRFO(Optimizer):
    """Boolean Manta Ray Foraging Optimization.

    Notes:
        Expresses chain, cyclone, and somersault foraging with Boolean operations.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the BMRFO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> BMRFO.")
        self.S = 1.0
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one BMRFO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position
        t = ctx.iteration / max(ctx.n_iterations, 1)

        for i in range(n):
            r1 = torch.rand(1, device=device).item()
            pos = pop.positions[i].bool()

            if r1 < 0.5:
                # Cyclone foraging (binary)
                r_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                beta_bin = torch.round(torch.rand_like(pop.positions[i])).bool()

                if t < torch.rand(1, device=device).item():
                    r_pos = torch.round(torch.rand_like(pop.positions[i])).bool()
                    ref = pop.positions[max(i - 1, 0)].bool() if i > 0 else r_pos
                    p1 = r_bin | (ref ^ pos)
                    p2 = beta_bin | (r_pos ^ pos)
                    new_pos = r_pos & p1 & p2
                else:
                    best_b = best.bool()
                    ref = pop.positions[max(i - 1, 0)].bool() if i > 0 else best_b
                    p1 = r_bin | (ref ^ pos)
                    p2 = beta_bin | (best_b ^ pos)
                    new_pos = best_b & p1 & p2
            else:
                # Chain foraging (binary)
                r_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                alpha_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                best_b = best.bool()

                if i == 0:
                    p1 = r_bin & (best_b ^ pos)
                    p2 = alpha_bin & (best_b ^ pos)
                else:
                    prev = pop.positions[i - 1].bool()
                    p1 = r_bin & (prev ^ pos)
                    p2 = alpha_bin & (best_b ^ pos)
                new_pos = pos | p1 | p2

            pop.positions[i] = new_pos.to(dtype=pop.dtype)

        pop.fitness = fn(pop.positions)
        pop.update_best()

        # Somersault foraging
        best_b = pop.best_position.bool()
        S_b = torch.tensor(self.S, device=device).bool() if self.S else torch.zeros(1, device=device).bool()

        for i in range(n):
            pos = pop.positions[i].bool()
            r1 = torch.round(torch.rand_like(pop.positions[i])).bool()
            r2 = torch.round(torch.rand_like(pop.positions[i])).bool()
            somersault = pos | (S_b & ((r1 ^ best_b) ^ (r2 ^ pos)))
            pop.positions[i] = somersault.to(dtype=pop.dtype)

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        pop.positions = pop.positions.clamp(min=lb, max=ub)


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

        logger.info("Overriding class: Optimizer -> BPSO.")
        self.c1 = 1.0
        self.c2 = 1.0
        super().__init__(params)
        logger.info("Class overrided.")

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


class UMDA(Optimizer):
    """Univariate Marginal Distribution Algorithm.

    Notes:
        Samples Boolean candidates from independently estimated marginal probabilities.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the UMDA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> UMDA.")
        self.p_selection = 0.75
        self.lower_bound_prob = 0.05
        self.upper_bound_prob = 0.95
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one UMDA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        n_selected = max(int(n * self.p_selection), 1)

        sorted_idx = torch.argsort(pop.fitness)
        selected = pop.positions[sorted_idx[:n_selected]]

        # Calculate probabilities
        probs = selected.mean(dim=0)
        probs = probs.clamp(min=self.lower_bound_prob, max=self.upper_bound_prob)

        # Sample new positions
        r = torch.rand_like(pop.positions)
        pop.positions = (probs.unsqueeze(0) > r).to(dtype=pop.dtype)
        pop.positions = pop.positions.clamp(min=lb, max=ub)
