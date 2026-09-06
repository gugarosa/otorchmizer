# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Black Widow Optimization.

References:
    V. Hayyolalam and A. A. Pourhaji Kazem.
    Black Widow Optimization Algorithm: A novel meta-heuristic approach
    for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2020).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BWO(Optimizer):
    """Black Widow Optimization.

    Notes:
        Mating, cannibalism, and mutation phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> BWO.")

        self.pp = 0.6
        self.cr = 0.44
        self.pm = 0.4

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def pp(self) -> float:
        """Return the procreation rate."""

        return self._pp

    @pp.setter
    def pp(self, pp: float) -> None:
        if not isinstance(pp, (float, int)):
            raise e.TypeError("`pp` must be a float or integer.")
        if not 0 <= pp <= 1:
            raise e.ValueError("`pp` must be between 0 and 1.")
        self._pp = pp

    @property
    def cr(self) -> float:
        """Return the cannibalism rate."""

        return self._cr

    @cr.setter
    def cr(self, cr: float) -> None:
        if not isinstance(cr, (float, int)):
            raise e.TypeError("`cr` must be a float or integer.")
        if not 0 <= cr <= 1:
            raise e.ValueError("`cr` must be between 0 and 1.")
        self._cr = cr

    @property
    def pm(self) -> float:
        """Return the mutation rate."""

        return self._pm

    @pm.setter
    def pm(self, pm: float) -> None:
        if not isinstance(pm, (float, int)):
            raise e.TypeError("`pm` must be a float or integer.")
        if not 0 <= pm <= 1:
            raise e.ValueError("`pm` must be between 0 and 1.")
        self._pm = pm

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        n_parents = max(int(n * self.pp), 2)
        if n_parents % 2 != 0:
            n_parents -= 1

        parent_idx = sorted_idx[:n_parents]
        fathers = pop.positions[parent_idx[: n_parents // 2]]
        mothers = pop.positions[parent_idx[n_parents // 2 :]]

        alpha = torch.rand(fathers.shape[0], 1, 1, device=device, dtype=pop.dtype)
        child1 = alpha * fathers + (1 - alpha) * mothers
        child2 = alpha * mothers + (1 - alpha) * fathers

        offspring = torch.cat([child1, child2], dim=0)
        offspring = offspring.clamp(min=lb, max=ub)
        offspring_fit = fn(offspring)

        n_survive = max(int(offspring.shape[0] * self.cr), 1)
        surv_idx = torch.argsort(offspring_fit)[:n_survive]
        survivors = offspring[surv_idx]
        surv_fit = offspring_fit[surv_idx]

        mutants = survivors.clone()
        mut_mask = torch.rand_like(mutants) < self.pm
        mutants = mutants + mut_mask.to(dtype=mutants.dtype) * torch.randn_like(mutants)
        mutants = mutants.clamp(min=lb, max=ub)
        mut_fit = fn(mutants)

        all_pos = torch.cat([pop.positions, survivors, mutants], dim=0)
        all_fit = torch.cat([pop.fitness, surv_fit, mut_fit], dim=0)

        best_idx = torch.argsort(all_fit)[:n]
        pop.positions = all_pos[best_idx]
        pop.fitness = all_fit[best_idx]
