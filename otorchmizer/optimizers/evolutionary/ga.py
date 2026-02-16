"""Genetic Algorithm — vectorized selection, crossover, and mutation.

References:
    M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GA(Optimizer):
    """Genetic Algorithm.

    Selection, crossover, and mutation are fully vectorized.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> GA.")

        self.p_selection = 0.75
        self.p_mutation = 0.25
        self.p_crossover = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def p_selection(self) -> float:
        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection: float) -> None:
        if not isinstance(p_selection, (float, int)):
            raise e.TypeError("`p_selection` should be a float or integer")
        self._p_selection = p_selection

    @property
    def p_mutation(self) -> float:
        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        self._p_mutation = p_mutation

    @property
    def p_crossover(self) -> float:
        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if not isinstance(p_crossover, (float, int)):
            raise e.TypeError("`p_crossover` should be a float or integer")
        self._p_crossover = p_crossover

    def _roulette_selection(self, population: Population) -> torch.Tensor:
        """Vectorized roulette selection.

        Returns indices of selected individuals.
        """

        n = population.n_agents
        n_selected = int(n * self.p_selection)
        if n_selected % 2 != 0:
            n_selected += 1
        n_selected = max(n_selected, 2)

        fitness = population.fitness
        max_fit = fitness.max()

        # Invert for minimization: f'(x) = f_max - f(x) + epsilon
        inv_fitness = max_fit - fitness + c.EPSILON
        probs = inv_fitness / inv_fitness.sum()

        selected = torch.multinomial(probs, n_selected, replacement=False)

        return selected

    def _crossover(self, parents_a: torch.Tensor,
                   parents_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized BLX crossover for all pairs simultaneously."""

        n_pairs = parents_a.shape[0]
        device = parents_a.device

        # Random blend factor per pair
        do_cross = torch.rand(n_pairs, 1, 1, device=device) < self.p_crossover
        r = torch.rand(n_pairs, 1, 1, device=device)

        alpha = torch.where(do_cross, r * parents_a + (1 - r) * parents_b, parents_a)
        beta = torch.where(do_cross, r * parents_b + (1 - r) * parents_a, parents_b)

        return alpha, beta

    def _mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Vectorized Gaussian mutation."""

        mask = torch.rand_like(offspring) < self.p_mutation
        noise = torch.randn_like(offspring)

        return offspring + mask.float() * noise

    def update(self, ctx: UpdateContext) -> None:
        """Vectorized GA: selection → crossover → mutation → evaluate → merge + sort."""

        pop = ctx.space.population
        function = ctx.function
        n = pop.n_agents

        selected = self._roulette_selection(pop)

        # Pair up selected individuals
        n_pairs = len(selected) // 2
        fathers = pop.positions[selected[:n_pairs]]
        mothers = pop.positions[selected[n_pairs:2 * n_pairs]]

        alpha, beta = self._crossover(fathers, mothers)
        alpha = self._mutation(alpha)
        beta = self._mutation(beta)

        # Clip offspring to bounds
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        alpha = alpha.clamp(min=lb, max=ub)
        beta = beta.clamp(min=lb, max=ub)

        # Evaluate offspring
        offspring = torch.cat([alpha, beta], dim=0)
        offspring_fit = function(offspring)

        # Merge with current population and keep the best n
        all_positions = torch.cat([pop.positions, offspring], dim=0)
        all_fitness = torch.cat([pop.fitness, offspring_fit], dim=0)

        sorted_idx = torch.argsort(all_fitness)[:n]
        pop.positions = all_positions[sorted_idx]
        pop.fitness = all_fitness[sorted_idx]
