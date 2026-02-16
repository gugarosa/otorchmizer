"""Parasitism-Predation Algorithm.

References:
    A. S. Mohamed, A. A. Hadi, and A. W. Mohamed.
    Parasitism – Predation algorithm (PPA).
    Soft Computing (2020).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.distribution as d
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PPA(Optimizer):
    """Parasitism-Predation Algorithm.

    Nesting (Lévy), parasitism (cuckoo), and predation (cat) phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> PPA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Dynamic population partitioning
        n_crows = max(round(n * (2 / 3 - t * (1 / 6))), 1)
        n_cats = max(round(n * (0.01 + t * (1 / 3 - 0.01))), 1)
        n_cuckoos = max(n - n_crows - n_cats, 0)

        sorted_idx = torch.argsort(pop.fitness)

        # --- Nesting Phase (Crows) ---
        crow_idx = sorted_idx[:n_crows]
        for ci in crow_idx:
            j = torch.randint(0, n, (1,), device=device).item()
            levy = d.generate_levy_distribution(beta=1.5, size=pop.positions[ci].shape, device=device)
            step = 0.01 * levy * (pop.positions[j] - pop.positions[ci])
            new_pos = pop.positions[ci] + step
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[ci]:
                pop.positions[ci] = new_pos
                pop.fitness[ci] = new_fit

        # --- Parasitism Phase (Cuckoos) ---
        if n_cuckoos > 0:
            cuckoo_idx = sorted_idx[n_crows:n_crows + n_cuckoos]
            p = t

            for ci in cuckoo_idx:
                j = torch.randint(0, n, (1,), device=device).item()
                S_g = (pop.positions[ci] - pop.positions[j]) * torch.rand(1, device=device)
                k = torch.bernoulli(torch.full(pop.positions[ci].shape, 1 - p, device=device))
                best_cuckoo = pop.positions[cuckoo_idx[0]]
                new_pos = best_cuckoo + S_g * k
                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]
                if new_fit < pop.fitness[ci]:
                    pop.positions[ci] = new_pos
                    pop.fitness[ci] = new_fit

        # --- Predation Phase (Cats) ---
        cat_idx = sorted_idx[n_crows + n_cuckoos:]
        constant = 2 - t

        for ci in cat_idx:
            r = torch.rand(1, 1, device=device)
            self.velocity[ci] = self.velocity[ci] + r * constant * (best.squeeze(0) - pop.positions[ci])
            new_pos = pop.positions[ci] + self.velocity[ci]
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[ci]:
                pop.positions[ci] = new_pos
                pop.fitness[ci] = new_fit
