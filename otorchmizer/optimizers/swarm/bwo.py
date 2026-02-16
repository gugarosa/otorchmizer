"""Black Widow Optimization.

References:
    V. Hayyolalam and A. A. Pourhaji Kazem.
    Black Widow Optimization Algorithm: A novel meta-heuristic approach
    for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2020).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BWO(Optimizer):
    """Black Widow Optimization.

    Mating, cannibalism, and mutation phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BWO.")

        self.pp = 0.6
        self.cr = 0.44
        self.pm = 0.4

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def pp(self) -> float:
        return self._pp

    @pp.setter
    def pp(self, pp: float) -> None:
        if not isinstance(pp, (float, int)):
            raise e.TypeError("`pp` should be a float or integer")
        if not 0 <= pp <= 1:
            raise e.ValueError("`pp` should be between 0 and 1")
        self._pp = pp

    @property
    def cr(self) -> float:
        return self._cr

    @cr.setter
    def cr(self, cr: float) -> None:
        if not isinstance(cr, (float, int)):
            raise e.TypeError("`cr` should be a float or integer")
        if not 0 <= cr <= 1:
            raise e.ValueError("`cr` should be between 0 and 1")
        self._cr = cr

    @property
    def pm(self) -> float:
        return self._pm

    @pm.setter
    def pm(self, pm: float) -> None:
        if not isinstance(pm, (float, int)):
            raise e.TypeError("`pm` should be a float or integer")
        if not 0 <= pm <= 1:
            raise e.ValueError("`pm` should be between 0 and 1")
        self._pm = pm

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Procreation: select top pp fraction as parents
        sorted_idx = torch.argsort(pop.fitness)
        n_parents = max(int(n * self.pp), 2)
        if n_parents % 2 != 0:
            n_parents -= 1

        parent_idx = sorted_idx[:n_parents]
        fathers = pop.positions[parent_idx[:n_parents // 2]]
        mothers = pop.positions[parent_idx[n_parents // 2:]]

        # Crossover
        alpha = torch.rand(fathers.shape[0], 1, 1, device=device)
        child1 = alpha * fathers + (1 - alpha) * mothers
        child2 = alpha * mothers + (1 - alpha) * fathers

        offspring = torch.cat([child1, child2], dim=0)
        offspring = offspring.clamp(min=lb, max=ub)
        offspring_fit = fn(offspring)

        # Cannibalism: keep only top cr fraction of offspring
        n_survive = max(int(offspring.shape[0] * self.cr), 1)
        surv_idx = torch.argsort(offspring_fit)[:n_survive]
        survivors = offspring[surv_idx]
        surv_fit = offspring_fit[surv_idx]

        # Mutation
        mutants = survivors.clone()
        mut_mask = torch.rand_like(mutants) < self.pm
        mutants = mutants + mut_mask.float() * torch.randn_like(mutants)
        mutants = mutants.clamp(min=lb, max=ub)
        mut_fit = fn(mutants)

        # Merge survivors + mutants + original, keep best n
        all_pos = torch.cat([pop.positions, survivors, mutants], dim=0)
        all_fit = torch.cat([pop.fitness, surv_fit, mut_fit], dim=0)

        best_idx = torch.argsort(all_fit)[:n]
        pop.positions = all_pos[best_idx]
        pop.fitness = all_fit[best_idx]
