# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Runner-Root Algorithm.

References:
    F. Merrikh-Bayat.
    The runner-root algorithm: A metaheuristic for solving unimodal
    and multimodal optimization problems.
    Applied Soft Computing (2015).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class RRA(Optimizer):
    """Apply runner and root movement with stall detection."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> RRA.")

        self.d_runner = 2.0
        self.d_root = 0.01
        self.tol = 0.01
        self.max_stall = 1000

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def d_runner(self) -> float:
        """Return the runner movement scale."""

        return self._d_runner

    @d_runner.setter
    def d_runner(self, d_runner: float) -> None:
        if not isinstance(d_runner, (float, int)):
            raise e.TypeError("`d_runner` must be a float or integer.")
        if d_runner <= 0:
            raise e.ValueError("`d_runner` must be positive.")
        self._d_runner = d_runner

    @property
    def d_root(self) -> float:
        """Return the root movement scale."""

        return self._d_root

    @d_root.setter
    def d_root(self, d_root: float) -> None:
        if not isinstance(d_root, (float, int)):
            raise e.TypeError("`d_root` must be a float or integer.")
        if d_root < 0:
            raise e.ValueError("`d_root` must be non-negative.")
        self._d_root = d_root

    @property
    def tol(self) -> float:
        """Return the relative-improvement tolerance."""

        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        if not isinstance(tol, (float, int)):
            raise e.TypeError("`tol` must be a float or integer.")
        if tol < 0:
            raise e.ValueError("`tol` must be non-negative.")
        self._tol = tol

    @property
    def max_stall(self) -> int:
        """Return the maximum consecutive stalled updates."""

        return self._max_stall

    @max_stall.setter
    def max_stall(self, max_stall: int) -> None:
        if not isinstance(max_stall, int):
            raise e.TypeError("`max_stall` must be an integer.")
        if max_stall <= 0:
            raise e.ValueError("`max_stall` must be positive.")
        self._max_stall = max_stall

    def compile(self, population) -> None:
        """Initialize stall tracking from the population best.

        Args:
            population: Population whose best fitness seeds the tracker.

        """

        self.n_stall = 0
        self.last_best_fit = population.best_fitness.clone()

    def _stalling_search(
        self,
        pop,
        position: torch.Tensor,
        fitness: torch.Tensor,
        function,
        is_large: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        best_position = position.clone()
        best_fitness = fitness.clone()

        for _ in range(max(pop.n_agents - 1, 0)):
            candidate = best_position.clone()
            variable = torch.randint(0, pop.n_variables, (1,), device=pop.device)
            if is_large:
                step = self.d_runner * torch.randn(1, device=pop.device, dtype=pop.dtype)
            else:
                step = self.d_root * (torch.rand(1, device=pop.device, dtype=pop.dtype) - 0.5)
            candidate[variable] += step
            candidate = candidate.clamp(min=pop.lb, max=pop.ub)
            candidate_fitness = function(candidate.unsqueeze(0))[0]
            if candidate_fitness < best_fitness:
                best_position = candidate
                best_fitness = candidate_fitness

        return best_position, best_fitness

    @staticmethod
    def _roulette_selection(fitness: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=fitness.device)

        shifted_fitness = fitness - fitness.min()
        inverse_fitness = 1 / (shifted_fitness + 0.1)
        probabilities = inverse_fitness / inverse_fitness.sum()
        return torch.multinomial(probabilities, n, replacement=True)

    def update(self, ctx: UpdateContext) -> None:
        """Generate daughters, apply stall searches, and reproduce the population.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        positions = pop.positions[sorted_idx]
        fitness = pop.fitness[sorted_idx]
        previous_best = fitness[0].clone()

        daughters = positions.clone()
        daughter_fit = fitness.clone()
        if pop.n_agents > 1:
            runner_step = self.d_runner * (
                torch.rand(
                    pop.n_agents - 1,
                    1,
                    1,
                    device=pop.device,
                    dtype=pop.dtype,
                )
                - 0.5
            )
            daughters[1:] = (daughters[1:] + runner_step).clamp(min=lb, max=ub)
            daughter_fit[1:] = fn(daughters[1:])

        daughter_idx = torch.argsort(daughter_fit)
        daughters = daughters[daughter_idx]
        daughter_fit = daughter_fit[daughter_idx]
        effectiveness = torch.abs(previous_best - daughter_fit[0]) / (previous_best.abs() + c.EPSILON)

        if effectiveness < self.tol:
            daughters[0], daughter_fit[0] = self._stalling_search(
                pop,
                daughters[0],
                daughter_fit[0],
                fn,
                is_large=True,
            )
            daughters[0], daughter_fit[0] = self._stalling_search(
                pop,
                daughters[0],
                daughter_fit[0],
                fn,
                is_large=False,
            )

        daughter_best = daughter_fit.argmin()
        if daughter_fit[daughter_best] < pop.best_fitness:
            pop.best_position = daughters[daughter_best].clone()
            pop.best_fitness = daughter_fit[daughter_best].clone()

        selected = self._roulette_selection(daughter_fit, pop.n_agents - 1)
        pop.positions = torch.cat((daughters[:1], daughters[selected]))
        pop.fitness = torch.cat((daughter_fit[:1], daughter_fit[selected]))

        effectiveness = torch.abs(previous_best - daughter_fit[0]) / (previous_best.abs() + c.EPSILON)
        self.n_stall = self.n_stall + 1 if effectiveness < self.tol else 0

        if self.n_stall >= self.max_stall:
            pop.positions = torch.rand_like(pop.positions) * (ub - lb) + lb
            pop.fitness = fn(pop.positions)
            restart_best = pop.fitness.argmin()
            if pop.fitness[restart_best] < pop.best_fitness:
                pop.best_position = pop.positions[restart_best].clone()
                pop.best_fitness = pop.fitness[restart_best].clone()
            self.n_stall = 0

        self.last_best_fit = pop.fitness.min().clone()
        pop.update_best()
