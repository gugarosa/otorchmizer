"""Jellyfish Search.

References:
    J.-S. Chou and D.-N. Truong.
    A novel metaheuristic optimizer inspired by behavior of jellyfish
    in ocean. Applied Mathematics and Computation (2021).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class JS(Optimizer):
    """Jellyfish Search optimizer.

    Ocean current, swarm motion, and time control mechanism.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> JS.")

        self.beta = 3.0
        self.gamma = 0.1
        self.eta = 4.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        self._beta = beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not isinstance(gamma, (float, int)):
            raise e.TypeError("`gamma` should be a float or integer")
        self._gamma = gamma

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, eta: float) -> None:
        if not isinstance(eta, (float, int)):
            raise e.TypeError("`eta` should be a float or integer")
        self._eta = eta

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Time control function
        c_t = torch.abs((1 - t) * (2 * torch.rand(1, device=device) - 1))

        if c_t.item() >= 0.5:
            # Ocean current
            mean_pos = pop.positions.mean(dim=0, keepdim=True)
            trend = best - self.beta * torch.rand(1, 1, 1, device=device) * mean_pos
            new_positions = pop.positions + torch.rand(n, 1, 1, device=device) * trend
        else:
            # Swarm motion
            r = torch.rand(n, device=device)
            j = torch.randint(0, n, (n,), device=device)

            active = (r >= 0.5).view(n, 1, 1)
            # Type A: move toward better neighbor
            direction_a = pop.positions[j] - pop.positions
            sign_a = torch.where(
                (pop.fitness[j] < pop.fitness).view(n, 1, 1),
                torch.ones(1, device=device),
                torch.ones(1, device=device) * -1,
            )
            step_a = self.gamma * torch.rand(n, 1, 1, device=device) * sign_a * direction_a

            # Type B: random position
            step_b = self.eta * torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb)

            new_positions = pop.positions + torch.where(active, step_a, step_b)

        pop.positions = new_positions.clamp(min=lb, max=ub)
