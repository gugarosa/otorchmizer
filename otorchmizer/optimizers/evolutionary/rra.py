"""Runner-Root Algorithm.

References:
    F. Merrikh-Bayat.
    The runner-root algorithm: A metaheuristic for solving unimodal
    and multimodal optimization problems.
    Applied Soft Computing (2015).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class RRA(Optimizer):
    """Runner-Root Algorithm.

    Runner and root movement with stall detection.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> RRA.")

        self.d_runner = 2.0
        self.d_root = 0.01
        self.tol = 0.01
        self.max_stall = 1000

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def d_runner(self) -> float:
        return self._d_runner

    @d_runner.setter
    def d_runner(self, d_runner: float) -> None:
        if not isinstance(d_runner, (float, int)):
            raise e.TypeError("`d_runner` should be a float or integer")
        self._d_runner = d_runner

    @property
    def d_root(self) -> float:
        return self._d_root

    @d_root.setter
    def d_root(self, d_root: float) -> None:
        if not isinstance(d_root, (float, int)):
            raise e.TypeError("`d_root` should be a float or integer")
        self._d_root = d_root

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        if not isinstance(tol, (float, int)):
            raise e.TypeError("`tol` should be a float or integer")
        self._tol = tol

    @property
    def max_stall(self) -> int:
        return self._max_stall

    @max_stall.setter
    def max_stall(self, max_stall: int) -> None:
        if not isinstance(max_stall, int):
            raise e.TypeError("`max_stall` should be an integer")
        if max_stall <= 0:
            raise e.ValueError("`max_stall` should be > 0")
        self._max_stall = max_stall

    def compile(self, population) -> None:
        self.n_stall = 0
        self.last_best_fit = population.best_fitness.clone()

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Runner movement
        daughters = pop.positions + self.d_runner * (torch.rand_like(pop.positions) - 0.5)
        daughters = daughters.clamp(min=lb, max=ub)
        daughter_fit = fn(daughters)

        # Check effectiveness
        best_fit = pop.best_fitness
        effectiveness = torch.abs(self.last_best_fit - best_fit) / (self.last_best_fit.abs() + c.EPSILON)

        if effectiveness < self.tol:
            # Large stalling search
            daughters = daughters + self.d_runner * torch.randn_like(daughters)
            daughters = daughters.clamp(min=lb, max=ub)
            daughter_fit = fn(daughters)

            # Small stalling (root) search
            roots = pop.positions + self.d_root * (torch.rand_like(pop.positions) - 0.5)
            roots = roots.clamp(min=lb, max=ub)
            root_fit = fn(roots)

            # Keep better of daughters vs roots
            use_root = root_fit < daughter_fit
            daughters[use_root] = roots[use_root]
            daughter_fit[use_root] = root_fit[use_root]

            self.n_stall += 1
        else:
            self.n_stall = 0

        # Replace if improved
        improved = daughter_fit < pop.fitness
        pop.positions[improved] = daughters[improved]
        pop.fitness[improved] = daughter_fit[improved]

        self.last_best_fit = pop.best_fitness.clone()

        # Stall reset
        if self.n_stall >= self.max_stall:
            pop.positions = torch.rand_like(pop.positions) * (ub - lb) + lb
            pop.fitness = fn(pop.positions)
            self.n_stall = 0
