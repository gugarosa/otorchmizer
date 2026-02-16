"""Gravitational Search Algorithm.

References:
    E. Rashedi, H. Nezamabadi-pour, and S. Saryazdi.
    GSA: a gravitational search algorithm.
    Information Sciences (2009).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GSA(Optimizer):
    """Gravitational Search Algorithm.

    Mass and force-based movement with decaying gravity.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> GSA.")

        self.G = 2.467

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def G(self) -> float:
        return self._G

    @G.setter
    def G(self, G: float) -> None:
        if not isinstance(G, (float, int)):
            raise e.TypeError("`G` should be a float or integer")
        if G < 0:
            raise e.ValueError("`G` should be >= 0")
        self._G = G

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.velocity = torch.zeros(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        t = ctx.iteration + 1
        G = self.G / t

        # Mass calculation
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        m = (pop.fitness - worst_fit) / (best_fit - worst_fit + c.EPSILON)
        M = m / (m.sum() + c.EPSILON)

        # Force calculation
        flat = pop.positions.reshape(n, -1)
        dist = torch.cdist(flat, flat).clamp(min=1e-10)

        force = torch.zeros_like(pop.positions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = torch.rand(1, device=device)
                    f = G * M[i] * M[j] / dist[i, j] * (pop.positions[j] - pop.positions[i])
                    force[i] += r * f

        # Acceleration
        accel = force / (M.view(n, 1, 1) + c.EPSILON)

        # Update velocity and position
        r = torch.rand(n, 1, 1, device=device)
        self.velocity = r * self.velocity + accel
        pop.positions = pop.positions + self.velocity
