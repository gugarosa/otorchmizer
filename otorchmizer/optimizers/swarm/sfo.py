"""Sailfish Optimizer.

References:
    S. Shadravan, H. R. Naji, and V. K. Bardsiri.
    The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
    for solving constrained engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SFO(Optimizer):
    """Sailfish Optimizer.

    Elite and sardine-based cooperative hunting.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SFO.")

        self.PP = 0.1
        self.A = 4.0
        self.e = 0.001

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def PP(self) -> float:
        return self._PP

    @PP.setter
    def PP(self, PP: float) -> None:
        if not isinstance(PP, (float, int)):
            raise e.TypeError("`PP` should be a float or integer")
        self._PP = PP

    @property
    def A(self) -> float:
        return self._A

    @A.setter
    def A(self, A: float) -> None:
        if not isinstance(A, (float, int)):
            raise e.TypeError("`A` should be a float or integer")
        self._A = A

    @property
    def e(self) -> float:
        return self._e

    @e.setter
    def e(self, e_val: float) -> None:
        if not isinstance(e_val, (float, int)):
            raise e.TypeError("`e` should be a float or integer")
        self._e = e_val

    def compile(self, population) -> None:
        # Sardines: bottom half of population
        n = population.n_agents
        self.n_sardines = n // 2
        self.n_sailfish = n - self.n_sardines
        self.sardine_positions = population.positions[self.n_sailfish:].clone()
        self.sardine_fitness = population.fitness[self.n_sailfish:].clone()

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Best sardine
        best_sardine_idx = self.sardine_fitness.argmin()
        best_sardine = self.sardine_positions[best_sardine_idx].unsqueeze(0)

        # Power decrease coefficient
        PD = 1 - (2 * ctx.iteration * self.e) / max(ctx.n_iterations, 1)
        PD = max(PD, 0.0)

        # Attack power
        AP = self.A * (1 - 2 * t * self.e)

        # Update sailfish positions
        r = torch.rand(n, 1, 1, device=device)
        new_positions = best - r * (best + best_sardine) / 2 - pop.positions

        new_positions = new_positions.clamp(min=lb, max=ub)
        new_fitness = fn(new_positions)

        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        # Update sardine positions
        n_s = self.sardine_positions.shape[0]
        alpha = torch.rand(n_s, pop.n_variables, pop.n_dimensions, device=device)
        mask = alpha < self.PP
        rand_pos = torch.rand_like(self.sardine_positions) * (ub - lb) + lb

        self.sardine_positions = torch.where(mask, rand_pos, self.sardine_positions + r[:n_s] * (best - self.sardine_positions + AP))
        self.sardine_positions = self.sardine_positions.clamp(min=lb, max=ub)
        self.sardine_fitness = fn(self.sardine_positions)
