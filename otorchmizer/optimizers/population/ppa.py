# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Parasitism-Predation Algorithm.

References:
    A. S. Mohamed, A. A. Hadi, and A. W. Mohamed.
    Parasitism – Predation algorithm (PPA).
    Soft Computing (2020).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.distribution as d
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PPA(Optimizer):
    """Apply crow nesting, cuckoo parasitism, and cat predation phases."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> PPA.")

        super().__init__(params)

        logger.info("Class overrided.")

    def compile(self, population) -> None:
        """Initialize the cat velocity state.

        Args:
            population: Population that defines state shape and device.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Partition the population and run the three search phases.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        n_crows = max(round(n * (2 / 3 - t * (1 / 6))), 1)
        n_cats = max(round(n * (0.01 + t * (1 / 3 - 0.01))), 1)
        n_cuckoos = max(n - n_crows - n_cats, 0)

        sorted_idx = torch.argsort(pop.fitness)

        # Crow nesting phase
        crow_idx = sorted_idx[:n_crows]
        for ci in crow_idx:
            j = torch.randint(0, n, (1,), device=device).item()
            levy = d.generate_levy_distribution(
                beta=1.5,
                size=pop.positions[ci].shape,
                device=device,
            ).to(pop.dtype)
            step = 0.01 * levy * (pop.positions[j] - pop.positions[ci])
            new_pos = pop.positions[ci] + step
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[ci]:
                pop.positions[ci] = new_pos
                pop.fitness[ci] = new_fit

        # Cuckoo parasitism phase
        if n_cuckoos > 0:
            cuckoo_idx = sorted_idx[n_crows : n_crows + n_cuckoos]
            p = t

            for ci in cuckoo_idx:
                j = torch.randint(0, n, (1,), device=device).item()
                S_g = (pop.positions[ci] - pop.positions[j]) * torch.rand(1, device=device, dtype=pop.dtype)
                k = torch.bernoulli(
                    torch.full(
                        pop.positions[ci].shape,
                        1 - p,
                        device=device,
                        dtype=pop.dtype,
                    )
                )
                best_cuckoo = pop.positions[cuckoo_idx[0]]
                new_pos = best_cuckoo + S_g * k
                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]
                if new_fit < pop.fitness[ci]:
                    pop.positions[ci] = new_pos
                    pop.fitness[ci] = new_fit

        # Cat predation phase
        cat_idx = sorted_idx[n_crows + n_cuckoos :]
        constant = 2 - t

        for ci in cat_idx:
            r = torch.rand(1, 1, device=device, dtype=pop.dtype)
            self.velocity[ci] = self.velocity[ci] + r * constant * (best.squeeze(0) - pop.positions[ci])
            new_pos = pop.positions[ci] + self.velocity[ci]
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[ci]:
                pop.positions[ci] = new_pos
                pop.fitness[ci] = new_fit
