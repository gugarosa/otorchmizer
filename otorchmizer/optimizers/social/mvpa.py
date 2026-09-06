# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Most Valuable Player Algorithm.

References:
    H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport.
    Operational Research (2017).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class MVPA(Optimizer):
    """Most Valuable Player Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the MVPA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_teams = 4
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if self.n_teams < 2:
            raise ValueError("`n_teams` must be at least 2.")
        if population.n_agents < self.n_teams:
            raise ValueError("`population.n_agents` must be at least `n_teams`.")

        self.n_p = population.n_agents // self.n_teams

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one MVPA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for team_i in range(self.n_teams):
            start_i = team_i * self.n_p
            end_i = start_i + self.n_p if team_i < self.n_teams - 1 else n

            team_pos = pop.positions[start_i:end_i]
            team_fit = pop.fitness[start_i:end_i]
            team_n = team_pos.shape[0]

            sorted_ti = torch.argsort(team_fit)
            franchise_i = team_pos[sorted_ti[0]].clone()
            fitness_i = team_fit.mean()

            # Select random opponent team
            j = torch.randint(0, self.n_teams, (1,), device=device).item()
            while j == team_i:
                j = torch.randint(0, self.n_teams, (1,), device=device).item()

            start_j = j * self.n_p
            end_j = start_j + self.n_p if j < self.n_teams - 1 else n
            team_j_fit = pop.fitness[start_j:end_j]
            team_j_pos = pop.positions[start_j:end_j]
            sorted_tj = torch.argsort(team_j_fit)
            franchise_j = team_j_pos[sorted_tj[0]].clone()
            fitness_j = team_j_fit.mean()

            for k in range(team_n):
                idx = start_i + k
                r1 = torch.rand(1, 1, device=device)
                r2 = torch.rand(1, device=device)
                r3 = torch.rand(1, 1, device=device)

                new_pos = (
                    pop.positions[idx]
                    + r1 * (franchise_i - pop.positions[idx])
                    + 2 * r1 * (best.squeeze(0) - pop.positions[idx])
                )

                Pr = 1 - fitness_i / (fitness_i + fitness_j + c.EPSILON)
                if r2.item() < Pr:
                    new_pos = new_pos + r3 * (pop.positions[idx] - franchise_j)
                else:
                    new_pos = new_pos + r3 * (franchise_j - pop.positions[idx])

                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]
                if new_fit < pop.fitness[idx]:
                    pop.positions[idx] = new_pos
                    pop.fitness[idx] = new_fit
