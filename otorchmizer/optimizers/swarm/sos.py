# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Symbiotic Organisms Search.

References:
    M.-Y. Cheng and D. Prayogo.
    Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
    Computers & Structures (2014).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SOS(Optimizer):
    """Symbiotic Organisms Search.

    Notes:
        Mutualism, commensalism, and parasitism phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

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
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for i in range(n):
            j = torch.randint(0, n, (1,), device=device).item()
            while j == i:
                j = torch.randint(0, n, (1,), device=device).item()

            mutual_vector = (pop.positions[i] + pop.positions[j]) / 2
            bf1 = torch.randint(1, 3, (1,), device=device).item()
            bf2 = torch.randint(1, 3, (1,), device=device).item()

            r1 = torch.rand(1, 1, device=device, dtype=pop.dtype)
            r2 = torch.rand(1, 1, device=device, dtype=pop.dtype)

            new_i = pop.positions[i] + r1 * (best.squeeze(0) - mutual_vector * bf1)
            new_j = pop.positions[j] + r2 * (best.squeeze(0) - mutual_vector * bf2)

            new_i = new_i.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_j = new_j.clamp(min=lb.squeeze(0), max=ub.squeeze(0))

            fit_i = fn(new_i.unsqueeze(0))[0]
            fit_j = fn(new_j.unsqueeze(0))[0]

            if fit_i < pop.fitness[i]:
                pop.positions[i] = new_i
                pop.fitness[i] = fit_i
            if fit_j < pop.fitness[j]:
                pop.positions[j] = new_j
                pop.fitness[j] = fit_j

            j = torch.randint(0, n, (1,), device=device).item()
            while j == i:
                j = torch.randint(0, n, (1,), device=device).item()

            r = torch.rand(1, 1, device=device, dtype=pop.dtype) * 2 - 1
            new_c = pop.positions[i] + r * (best.squeeze(0) - pop.positions[j])
            new_c = new_c.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            fit_c = fn(new_c.unsqueeze(0))[0]

            if fit_c < pop.fitness[i]:
                pop.positions[i] = new_c
                pop.fitness[i] = fit_c

            j = torch.randint(0, n, (1,), device=device).item()
            while j == i:
                j = torch.randint(0, n, (1,), device=device).item()

            parasite = pop.positions[i].clone()

            n_dims_to_change = torch.randint(1, pop.n_variables + 1, (1,), device=device).item()
            dims = torch.randperm(pop.n_variables, device=device)[:n_dims_to_change]
            parasite[dims] = (
                torch.rand(n_dims_to_change, pop.n_dimensions, device=device, dtype=pop.dtype)
                * (ub.squeeze(0)[dims] - lb.squeeze(0)[dims])
                + lb.squeeze(0)[dims]
            )

            fit_p = fn(parasite.unsqueeze(0))[0]
            if fit_p < pop.fitness[j]:
                pop.positions[j] = parasite
                pop.fitness[j] = fit_p
