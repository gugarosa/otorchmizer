# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Genetic Algorithm — vectorized selection, crossover, and mutation.

References:
    M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class GA(Optimizer):
    """Apply vectorized selection, crossover, and mutation with a Genetic Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.p_selection = 0.75
        self.p_mutation = 0.25
        self.p_crossover = 0.5

        super().__init__(params)

    @property
    def p_selection(self) -> float:
        """Return the parent-selection ratio."""

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection: float) -> None:
        if not isinstance(p_selection, Real):
            raise TypeError("`p_selection` must be a float or integer.")
        if not 0 <= p_selection <= 1:
            raise ValueError("`p_selection` must be between 0 and 1.")
        self._p_selection = float(p_selection)

    @property
    def p_mutation(self) -> float:
        """Return the mutation probability."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, Real):
            raise TypeError("`p_mutation` must be a float or integer.")
        if not 0 <= p_mutation <= 1:
            raise ValueError("`p_mutation` must be between 0 and 1.")
        self._p_mutation = float(p_mutation)

    @property
    def p_crossover(self) -> float:
        """Return the crossover probability."""

        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if not isinstance(p_crossover, Real):
            raise TypeError("`p_crossover` must be a float or integer.")
        if not 0 <= p_crossover <= 1:
            raise ValueError("`p_crossover` must be between 0 and 1.")
        self._p_crossover = float(p_crossover)

    def _roulette_selection(self, population: Population) -> torch.Tensor:
        n = population.n_agents
        n_selected = int(n * self.p_selection)
        if n_selected % 2 != 0:
            n_selected += 1
        n_selected = min(max(n_selected, 2), n - n % 2)

        if n_selected == 0:
            return torch.empty(0, dtype=torch.long, device=population.device)

        fitness = population.fitness
        max_fit = fitness.max()

        # Invert for minimization: f'(x) = f_max - f(x) + epsilon
        inv_fitness = max_fit - fitness + c.EPSILON
        probs = inv_fitness / inv_fitness.sum()

        selected = torch.multinomial(probs, n_selected, replacement=False)

        return selected

    def _crossover(self, parents_a: torch.Tensor, parents_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_pairs = parents_a.shape[0]
        device = parents_a.device

        do_cross = torch.rand(n_pairs, 1, 1, device=device, dtype=parents_a.dtype) < self.p_crossover
        r = torch.rand(n_pairs, 1, 1, device=device, dtype=parents_a.dtype)

        alpha = torch.where(do_cross, r * parents_a + (1 - r) * parents_b, parents_a)
        beta = torch.where(do_cross, r * parents_b + (1 - r) * parents_a, parents_b)

        return alpha, beta

    def _mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(offspring) < self.p_mutation
        noise = torch.randn_like(offspring)

        return offspring + mask.to(offspring.dtype) * noise

    def update(self, ctx: UpdateContext) -> None:
        """Select parents, create offspring, and retain the best candidates.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        function = ctx.function
        n = pop.n_agents

        selected = self._roulette_selection(pop)
        if selected.numel() == 0:
            return

        n_pairs = len(selected) // 2
        fathers = pop.positions[selected[:n_pairs]]
        mothers = pop.positions[selected[n_pairs : 2 * n_pairs]]

        alpha, beta = self._crossover(fathers, mothers)
        alpha = self._mutation(alpha)
        beta = self._mutation(beta)

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        alpha = alpha.clamp(min=lb, max=ub)
        beta = beta.clamp(min=lb, max=ub)

        offspring = torch.cat([alpha, beta], dim=0)
        offspring_fit = function(offspring)

        all_positions = torch.cat([pop.positions, offspring], dim=0)
        all_fitness = torch.cat([pop.fitness, offspring_fit], dim=0)

        sorted_idx = torch.argsort(all_fitness)[:n]
        pop.positions = all_positions[sorted_idx]
        pop.fitness = all_fitness[sorted_idx]
