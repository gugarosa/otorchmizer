# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Queuing Search Algorithm.

References:
    J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems.
    Applied Mathematical Modelling (2018).

"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Any

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class QSA(Optimizer):
    """Queuing Search Algorithm.

    Notes:
        Runs all three business phases with greedy acceptance.
        Queue sizes follow reciprocal positive leader fitness, with equal shares for nonpositive leaders.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the QSA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population) -> None:
        """Validate the population required by the three queue leaders.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            ValueError: If fewer than three agents are available.

        """

        if population.n_agents < 3:
            raise ValueError("`population.n_agents` must be at least 3 for QSA.")

    def _sort_queues(self, population: Population) -> tuple[torch.Tensor, int, int]:
        population.sort_by_fitness()
        fitness = population.fitness[:3]
        if fitness[0] > 0:
            weights = fitness[0] / fitness
            weights /= weights.sum()
        else:
            weights = torch.full_like(fitness, 1 / 3)

        first = int((weights[0] * population.n_agents).item())
        second = first + int((weights[1] * population.n_agents).item())
        return population.positions[:3].clone(), first, second

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one QSA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        Raises:
            ValueError: The population contains non-finite fitness values.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        if not torch.isfinite(pop.fitness).all():
            raise ValueError("`population.fitness` must be finite for QSA queue allocation.")
        progress = ctx.iteration / max(ctx.n_iterations, 1)
        beta = exp(log(1 / max(ctx.iteration, c.EPSILON)) * sqrt(progress))
        shape = (pop.n_variables, pop.n_dimensions)

        leaders, first, second = self._sort_queues(pop)
        case = 1
        for i in range(n):
            if i in (0, first, second):
                case = 1
            leader = leaders[0 if i < first else 1 if i < second else 2]
            alpha = 2 * torch.rand((), device=device, dtype=pop.dtype) - 1
            energy = r.generate_gamma_random_number(1, 0.5, shape, device=device, dtype=pop.dtype)
            fluctuation = beta * alpha * energy * (leader - pop.positions[i]).abs()
            if case == 1:
                jitter = r.generate_gamma_random_number(1, 0.5, 1, device=device, dtype=pop.dtype)
                candidate = leader + fluctuation + jitter * (leader - pop.positions[i])
            else:
                candidate = pop.positions[i] + fluctuation

            candidate = candidate.clamp(min=pop.lb, max=pop.ub)
            fitness = fn(candidate.unsqueeze(0))[0]
            if fitness < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = fitness
            else:
                case = 3 - case
        pop.update_best()

        leaders, first, second = self._sort_queues(pop)
        leader_fitness = pop.fitness[:3] / pop.fitness[:3].abs().max().clamp_min(torch.finfo(pop.dtype).tiny)
        denominator = leader_fitness[1] + leader_fitness[2]
        cv = torch.where(denominator != 0, leader_fitness[0] / denominator, torch.zeros_like(denominator))
        cv = cv.clamp(0, 1)
        for i in range(n):
            if torch.rand((), device=device, dtype=pop.dtype) >= (i + 1) / n:
                continue
            leader = leaders[0 if i < first else 1 if i < second else 2]
            donors = pop.positions[torch.randperm(n, device=device)[:2]]
            coin = torch.rand((), device=device, dtype=pop.dtype)
            jitter = r.generate_gamma_random_number(1, 0.5, 1, device=device, dtype=pop.dtype)
            direction = donors[0] - donors[1] if coin < cv else leader - donors[0]
            candidate = (pop.positions[i] + jitter * direction).clamp(min=pop.lb, max=pop.ub)
            fitness = fn(candidate.unsqueeze(0))[0]
            if fitness < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = fitness
        pop.update_best()

        pop.sort_by_fitness()
        for i in range(n):
            candidate = pop.positions[i].clone()
            for variable in range(pop.n_variables):
                if torch.rand((), device=device, dtype=pop.dtype) >= (i + 1) / n:
                    continue
                donors = pop.positions[torch.randperm(n, device=device)[:2]]
                jitter = r.generate_gamma_random_number(1, 0.5, 1, device=device, dtype=pop.dtype)
                candidate[variable] = donors[0, variable] + jitter * (donors[1, variable] - candidate[variable])
                candidate = candidate.clamp(min=pop.lb, max=pop.ub)
                fitness = fn(candidate.unsqueeze(0))[0]
                if fitness < pop.fitness[i]:
                    pop.positions[i] = candidate
                    pop.fitness[i] = fitness
        pop.update_best()
