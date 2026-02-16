"""Atom Search Optimization.

References:
    W. Zhao, L. Wang, and Z. Zhang.
    Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem.
    Knowledge-Based Systems (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ASO(Optimizer):
    """Atom Search Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> ASO.")
        self.alpha = 50.0
        self.beta = 0.2
        super().__init__(params)
        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        self._beta = beta

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        G = torch.exp(torch.tensor(-20.0 * t, device=device))

        # Mass
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        m = torch.exp(-(pop.fitness - best_fit) / (worst_fit - best_fit + c.EPSILON))
        M = m / (m.sum() + c.EPSILON)

        # K best
        K = max(int(n * (1 - t)), 2)
        sorted_idx = torch.argsort(pop.fitness)[:K]

        # Acceleration
        accel = torch.zeros_like(pop.positions)
        for i in range(n):
            for j_idx in sorted_idx:
                if j_idx == i:
                    continue
                diff = pop.positions[j_idx] - pop.positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                r = torch.rand(1, device=device)
                accel[i] += r * G * M[j_idx] * diff / dist

        r = torch.rand(n, 1, 1, device=device)
        self.velocity = r * self.velocity + accel + self.beta * (best - pop.positions)
        pop.positions = pop.positions + self.velocity
