"""Simulated Annealing.

References:
    S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi.
    Optimization by simulated annealing.
    Science (1983).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SA(Optimizer):
    """Simulated Annealing.

    Temperature-controlled Metropolis-Hastings acceptance.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SA.")

        self.T = 100.0
        self.beta = 0.999

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, T: float) -> None:
        if not isinstance(T, (float, int)):
            raise e.TypeError("`T` should be a float or integer")
        self._T = T

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        self._beta = beta

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        # Add Gaussian noise
        noise = torch.randn_like(pop.positions) * 0.1
        new_positions = pop.positions + noise
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        # Metropolis acceptance criterion
        delta = new_fitness - pop.fitness
        accept_prob = torch.exp(-delta / max(self.T, 1e-10))
        r = torch.rand(n, device=device)
        accept = (new_fitness < pop.fitness) | (r < accept_prob)

        pop.positions[accept] = new_positions[accept]
        pop.fitness[accept] = new_fitness[accept]

        # Cool temperature
        self.T *= self.beta
