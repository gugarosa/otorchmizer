"""Boolean-based optimizers: BMRFO, BPSO, UMDA."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BMRFO(Optimizer):
    """Boolean Manta Ray Foraging Optimization.

    Binary chain, cyclone, and somersault foraging with XOR/AND/OR logic.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BMRFO.")
        self.S = 1.0
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
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

            pop.positions[i] = new_pos.float()

        # Evaluate
        pop.fitness = fn(pop.positions)

        # Somersault foraging
        best_b = pop.best_position.bool()
        S_b = torch.tensor(self.S, device=device).bool() if self.S else torch.zeros(1, device=device).bool()

        for i in range(n):
            pos = pop.positions[i].bool()
            r1 = torch.round(torch.rand_like(pop.positions[i])).bool()
            r2 = torch.round(torch.rand_like(pop.positions[i])).bool()
            somersault = pos | (S_b & ((r1 ^ best_b) ^ (r2 ^ pos)))
            pop.positions[i] = somersault.float()

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        pop.positions = pop.positions.clamp(min=lb, max=ub)


class BPSO(Optimizer):
    """Boolean Particle Swarm Optimization.

    Binary PSO with XOR-based velocity and position updates.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BPSO.")
        self.c1 = 1.0
        self.c2 = 1.0
        super().__init__(params)
        logger.info("Class overrided.")

    def compile(self, population) -> None:
        n = population.n_agents
        shape = (n, population.n_variables, population.n_dimensions)
        device = population.device
        self.local_position = torch.zeros(shape, dtype=torch.bool, device=device)
        self.velocity = torch.zeros(shape, dtype=torch.bool, device=device)

    def evaluate(self, population, function) -> None:
        population.fitness = function(population.positions)
        for i in range(population.n_agents):
            if population.fitness[i] < population.best_fitness:
                self.local_position[i] = population.positions[i].bool()
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
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
            pop.positions[i] = (pos ^ self.velocity[i]).float()


class UMDA(Optimizer):
    """Univariate Marginal Distribution Algorithm.

    Probability-based binary optimization.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> UMDA.")
        self.p_selection = 0.75
        self.lower_bound_prob = 0.05
        self.upper_bound_prob = 0.95
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
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
        pop.positions = (probs.unsqueeze(0) > r).float()
        pop.positions = pop.positions.clamp(min=lb, max=ub)
