"""Bat Algorithm.

References:
    X.-S. Yang. A new metaheuristic bat-inspired algorithm.
    Nature Inspired Cooperative Strategies for Optimization (2010).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BA(Optimizer):
    """Bat Algorithm.

    Vectorized echolocation-based search using frequency, velocity, and loudness.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> BA.")

        self.f_min = 0.0
        self.f_max = 2.0
        self.A = 0.5
        self.r = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def f_min(self) -> float:
        return self._f_min

    @f_min.setter
    def f_min(self, f_min: float) -> None:
        if not isinstance(f_min, (float, int)):
            raise e.TypeError("`f_min` should be a float or integer")
        self._f_min = f_min

    @property
    def f_max(self) -> float:
        return self._f_max

    @f_max.setter
    def f_max(self, f_max: float) -> None:
        if not isinstance(f_max, (float, int)):
            raise e.TypeError("`f_max` should be a float or integer")
        self._f_max = f_max

    @property
    def A(self) -> float:
        return self._A

    @A.setter
    def A(self, A: float) -> None:
        if not isinstance(A, (float, int)):
            raise e.TypeError("`A` should be a float or integer")
        self._A = A

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not isinstance(r, (float, int)):
            raise e.TypeError("`r` should be a float or integer")
        self._r = r

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device)
        self.frequency = torch.zeros(population.n_agents, device=population.device)
        self.loudness = torch.full((population.n_agents,), self.A, device=population.device)
        self.pulse_rate = torch.full((population.n_agents,), self.r, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        best = pop.best_position.unsqueeze(0)

        # Update frequency, velocity, position
        beta = torch.rand(n, 1, 1, device=device)
        self.frequency = self.f_min + (self.f_max - self.f_min) * beta.squeeze()
        self.velocity = self.velocity + (pop.positions - best) * self.frequency.view(n, 1, 1)
        new_positions = pop.positions + self.velocity

        # Local search for high pulse rate
        r_test = torch.rand(n, device=device)
        local_mask = r_test > self.pulse_rate
        if local_mask.any():
            mean_loud = self.loudness.mean()
            noise = torch.randn_like(new_positions[local_mask]) * mean_loud
            new_positions[local_mask] = best + noise

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        # Accept if better and random < loudness
        r_accept = torch.rand(n, device=device)
        accept = (new_fitness < pop.fitness) & (r_accept < self.loudness)

        pop.positions[accept] = new_positions[accept]
        pop.fitness[accept] = new_fitness[accept]

        # Update loudness and pulse rate for accepted
        self.loudness[accept] *= 0.9
        self.pulse_rate[accept] = self.r * (1 - torch.exp(torch.tensor(-0.9 * (ctx.iteration + 1), device=device)))
