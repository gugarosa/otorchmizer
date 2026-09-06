# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Slime Mould Algorithm.

References:
    S. Li, H. Chen, M. Wang, A. A. Heidari, S. Mirjalili
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems (2020).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SMA(Optimizer):
    """Slime Mould Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SMA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.z = 0.03
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.weight = population.positions.new_ones(shape)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one SMA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]
        fit_range = (worst_fit - best_fit).clamp_min(torch.finfo(pop.fitness.dtype).eps)

        # Update weights
        for rank, idx in enumerate(sorted_idx):
            r = torch.rand_like(self.weight[idx])
            log_val = torch.log10((pop.fitness[idx] - best_fit) / fit_range + 1)
            if rank < n // 2:
                self.weight[idx] = 1 + r * log_val
            else:
                self.weight[idx] = 1 - r * log_val

        a_val = torch.atanh(torch.tensor(-(t + 1) / (max(ctx.n_iterations, 1) + 1) + 1, device=device)).clamp(max=5)
        b_val = 1 - (t + 1) / (max(ctx.n_iterations, 1) + 1)

        for i in range(n):
            r = torch.rand(1, device=device).item()
            if r < self.z:
                pop.positions[i] = torch.rand_like(pop.positions[i]) * (ub.squeeze(0) - lb.squeeze(0)) + lb.squeeze(0)
            else:
                p = torch.tanh(torch.abs(pop.fitness[i] - best_fit))
                vb = torch.rand_like(pop.positions[i]) * 2 * a_val - a_val
                vc = torch.rand_like(pop.positions[i]) * 2 * b_val - b_val

                if torch.rand(1, device=device).item() < p.item():
                    k = torch.randint(0, n, (1,), device=device).item()
                    l_idx = torch.randint(0, n, (1,), device=device).item()
                    pop.positions[i] = best.squeeze(0) + vb * self.weight[i] * (pop.positions[k] - pop.positions[l_idx])
                else:
                    pop.positions[i] = pop.positions[i] * vc

        pop.positions = pop.positions.clamp(min=lb, max=ub)
