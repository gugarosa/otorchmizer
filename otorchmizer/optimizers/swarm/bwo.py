# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Black Widow Optimization.

References:
    V. Hayyolalam and A. A. Pourhaji Kazem.
    Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2020).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BWO(Optimizer):
    """Black Widow Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.pp = 0.6
        self.cr = 0.44
        self.pm = 0.4

        super().__init__(params)

    @property
    def pp(self) -> float:
        """Return the procreation proportion."""

        return self._pp

    @pp.setter
    def pp(self, pp: float) -> None:
        self._pp = self._validate_probability("pp", pp)

    @property
    def cr(self) -> float:
        """Return the cannibal survival proportion."""

        return self._cr

    @cr.setter
    def cr(self, cr: float) -> None:
        self._cr = self._validate_probability("cr", cr)

    @property
    def pm(self) -> float:
        """Return the mutation proportion."""

        return self._pm

    @pm.setter
    def pm(self, pm: float) -> None:
        self._pm = self._validate_probability("pm", pm)

    @staticmethod
    def _validate_probability(name: str, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(f"`{name}` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError(f"`{name}` must be between 0 and 1.")
        return value

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one reproduction, cannibalism, and mutation cycle.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        n_reproduce = int(pop.n_agents * self.pp)
        n_cannibals = max(int(pop.n_agents * self.cr), 1) if self.cr > 0 else 0
        n_mutate = int(pop.n_agents * self.pm)
        order = torch.argsort(pop.fitness)
        mutation_pool = pop.positions[order[:n_reproduce]].clone()
        generated_positions = []
        generated_fitness = []

        for _ in range(n_reproduce):
            parents = torch.randint(0, pop.n_agents, (2,), device=pop.device)
            father = pop.positions[parents[0]]
            mother = pop.positions[parents[1]]
            family_positions = []
            family_fitness = []

            for _ in range(max(pop.n_variables // 2, 1)):
                alpha = torch.rand((), device=pop.device, dtype=pop.dtype)
                child1 = alpha * father + (1 - alpha) * mother
                child2 = alpha * mother + (1 - alpha) * father
                children = torch.stack((mother, child1, child2)).clamp(
                    min=pop.lb.unsqueeze(0),
                    max=pop.ub.unsqueeze(0),
                )
                family_positions.append(children)
                fitness = ctx.function(children)
                family_fitness.append(fitness)
                self._update_archive(pop, children, fitness)

            if family_positions and n_cannibals:
                positions = torch.cat(family_positions)
                fitness = torch.cat(family_fitness)
                survivors = torch.argsort(fitness)[:n_cannibals]
                generated_positions.append(positions[survivors])
                generated_fitness.append(fitness[survivors])

        if n_reproduce and pop.n_variables > 1:
            for _ in range(n_mutate):
                mutant = mutation_pool[torch.randint(0, n_reproduce, (), device=pop.device)].clone()
                variables = torch.randperm(pop.n_variables, device=pop.device)[:2]
                first = mutant[variables[0]].clone()
                mutant[variables[0]] = mutant[variables[1]]
                mutant[variables[1]] = first
                mutant = mutant.clamp(min=pop.lb, max=pop.ub)
                mutant_fitness = ctx.function(mutant.unsqueeze(0))
                generated_positions.append(mutant.unsqueeze(0))
                generated_fitness.append(mutant_fitness)
                self._update_archive(pop, mutant.unsqueeze(0), mutant_fitness)

        if generated_positions:
            all_positions = torch.cat((pop.positions, *generated_positions))
            all_fitness = torch.cat((pop.fitness, *generated_fitness))
            selected = torch.argsort(all_fitness)[: pop.n_agents]
            pop.positions = all_positions[selected]
            pop.fitness = all_fitness[selected]
            pop.update_best()

    @staticmethod
    def _update_archive(population, positions: torch.Tensor, fitness: torch.Tensor) -> None:
        best_idx = fitness.argmin()
        if fitness[best_idx] < population.best_fitness:
            population.best_fitness = fitness[best_idx].clone()
            population.best_position = positions[best_idx].clone()
