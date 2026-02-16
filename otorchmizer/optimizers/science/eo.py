"""Equilibrium Optimizer.

References:
    A. Faramarzi et al.
    Equilibrium optimizer: A novel optimization algorithm.
    Knowledge-Based Systems (2020).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class EO(Optimizer):
    """Equilibrium Optimizer.

    Concentration-based update with generation control.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> EO.")

        self.a1 = 2.0
        self.a2 = 1.0
        self.GP = 0.5
        self.V = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def a1(self) -> float:
        return self._a1

    @a1.setter
    def a1(self, a1: float) -> None:
        if not isinstance(a1, (float, int)):
            raise e.TypeError("`a1` should be a float or integer")
        self._a1 = a1

    @property
    def a2(self) -> float:
        return self._a2

    @a2.setter
    def a2(self, a2: float) -> None:
        if not isinstance(a2, (float, int)):
            raise e.TypeError("`a2` should be a float or integer")
        self._a2 = a2

    @property
    def GP(self) -> float:
        return self._GP

    @GP.setter
    def GP(self, GP: float) -> None:
        if not isinstance(GP, (float, int)):
            raise e.TypeError("`GP` should be a float or integer")
        if not 0 <= GP <= 1:
            raise e.ValueError("`GP` should be between 0 and 1")
        self._GP = GP

    def compile(self, population) -> None:
        # Maintain top-4 equilibrium candidates
        shape = (population.n_variables, population.n_dimensions)
        device = population.device
        self.C = [
            torch.zeros(shape, device=device) for _ in range(4)
        ]
        self.C_fit = [torch.tensor(c.FLOAT_MAX, device=device) for _ in range(4)]

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        # Update equilibrium pool (top-4)
        for i in range(n):
            for k in range(4):
                if pop.fitness[i] < self.C_fit[k]:
                    # Shift down
                    for j in range(3, k, -1):
                        self.C[j] = self.C[j - 1].clone()
                        self.C_fit[j] = self.C_fit[j - 1].clone()
                    self.C[k] = pop.positions[i].clone()
                    self.C_fit[k] = pop.fitness[i].clone()
                    break

        # Average concentration
        C_avg = sum(self.C) / 4
        C_pool = self.C + [C_avg]

        # Time factor
        time = (1 - t / T) ** (self.a2 * t / T)

        for i in range(n):
            # Random equilibrium from pool
            idx = torch.randint(0, 5, (1,)).item()
            C_eq = C_pool[idx]

            r = torch.rand_like(pop.positions[i])
            lam = torch.rand_like(pop.positions[i])

            # Exponential term
            F = self.a1 * torch.sign(r - 0.5) * (torch.exp(-lam * time) - 1)

            # Generation probability
            r_GP = torch.rand(1, device=device)
            GCP = 0.5 * r_GP if r_GP >= self.GP else torch.tensor(0.0, device=device)

            r1 = torch.rand_like(pop.positions[i])
            r2 = torch.rand_like(pop.positions[i])
            G = GCP * (C_eq - lam * pop.positions[i])

            pop.positions[i] = C_eq + (pop.positions[i] - C_eq) * F + (G / (lam * self.V + 1e-10)) * (1 - F)

        pop.positions = pop.positions.clamp(min=lb, max=ub)
