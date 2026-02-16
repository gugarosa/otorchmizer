"""Sooty Tern Optimization Algorithm.

References:
    G. Dhiman and A. Kaur.
    STOA: A bio-inspired based optimization algorithm for industrial
    engineering problems.
    Engineering Applications of Artificial Intelligence (2019).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class STOA(Optimizer):
    """Sooty Tern Optimization Algorithm.

    Collision avoidance, convergence, and attack phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> STOA.")

        self.Cf = 2.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def Cf(self) -> float:
        return self._Cf

    @Cf.setter
    def Cf(self, Cf: float) -> None:
        if not isinstance(Cf, (float, int)):
            raise e.TypeError("`Cf` should be a float or integer")
        self._Cf = Cf

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Linearly decreasing Sa from Cf to 0
        Sa = self.Cf - t * self.Cf

        # Collision avoidance
        Cb = 0.5 * torch.rand(n, 1, 1, device=device)
        diff = Sa * (best - torch.rand(n, 1, 1, device=device) * pop.positions)

        # Convergence
        M = Cb * diff

        # Attack: spiral
        k = torch.rand(n, 1, 1, device=device) * 2 * torch.pi
        r_spiral = torch.rand(n, 1, 1, device=device)

        x = r_spiral * torch.sin(k)
        y = r_spiral * torch.cos(k)
        z = r_spiral * k

        pop.positions = M * (x + y + z) + best
