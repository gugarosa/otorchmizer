# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Walrus Optimization Algorithm.

References:
    P. Trojovský and M. Dehghani.
    A new bio-inspired metaheuristic algorithm for solving optimization problems based on walruses behavior.
    Scientific Reports (2023).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class WAOA(Optimizer):
    """Walrus Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one feeding, migration, and exploration cycle.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        best = pop.best_position.unsqueeze(0)

        integers = torch.randint(
            1,
            3,
            pop.positions.shape,
            device=pop.device,
        )
        random = torch.rand_like(pop.positions)
        candidates = pop.positions + random * (best - integers * pop.positions)
        self._accept(pop, fn, candidates.clamp(min=lb, max=ub))

        if pop.n_agents > 1:
            for i in range(pop.n_agents):
                partner = torch.randint(0, pop.n_agents - 1, (), device=pop.device)
                partner += (partner >= i).long()
                random = torch.rand_like(pop.positions[i])
                if pop.fitness[partner] < pop.fitness[i]:
                    integers = torch.randint(
                        1,
                        3,
                        pop.positions[i].shape,
                        device=pop.device,
                    )
                    candidate = pop.positions[i] + random * (pop.positions[partner] - integers * pop.positions[i])
                else:
                    candidate = pop.positions[i] + random * (pop.positions[i] - pop.positions[partner])
                candidate = candidate.clamp(min=pop.lb, max=pop.ub)
                candidate_fitness = fn(candidate.unsqueeze(0))[0]
                if candidate_fitness < pop.fitness[i]:
                    pop.positions[i] = candidate
                    pop.fitness[i] = candidate_fitness

        iteration = ctx.iteration + 1
        local_lb = pop.lb.unsqueeze(0) / iteration
        local_ub = pop.ub.unsqueeze(0) / iteration
        random = torch.rand_like(pop.positions)
        candidates = pop.positions + local_lb + random * (local_ub - local_lb)
        self._accept(pop, fn, candidates.clamp(min=lb, max=ub))
        pop.update_best()

    @staticmethod
    def _accept(population, function, candidates: torch.Tensor) -> None:
        candidate_fitness = function(candidates)
        improved = candidate_fitness < population.fitness
        population.positions[improved] = candidates[improved]
        population.fitness[improved] = candidate_fitness[improved]
