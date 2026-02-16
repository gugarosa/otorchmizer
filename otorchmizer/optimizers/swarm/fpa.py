"""Flower Pollination Algorithm.

References:
    X.-S. Yang.
    Flower pollination algorithm for global optimization.
    Unconventional Computation and Natural Computation (2012).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FPA(Optimizer):
    """Flower Pollination Algorithm.

    Global (Lévy) and local pollination phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> FPA.")

        self.beta = 1.5
        self.eta = 0.2
        self.p = 0.8

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
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, eta: float) -> None:
        if not isinstance(eta, (float, int)):
            raise e.TypeError("`eta` should be a float or integer")
        self._eta = eta

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        if not 0 <= p <= 1:
            raise e.ValueError("`p` should be between 0 and 1")
        self._p = p

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        r = torch.rand(n, device=device)

        # Global pollination (Lévy flights)
        levy = d.generate_levy_distribution(
            beta=self.beta,
            size=pop.positions.shape,
            device=device,
        )
        global_pos = pop.positions + self.eta * levy * (best - pop.positions)

        # Local pollination
        j = torch.randint(0, n, (n,), device=device)
        k = torch.randint(0, n, (n,), device=device)
        epsilon = torch.rand(n, 1, 1, device=device)
        local_pos = pop.positions + epsilon * (pop.positions[j] - pop.positions[k])

        # Select based on probability
        use_global = (r < self.p).view(n, 1, 1)
        new_positions = torch.where(use_global, global_pos, local_pos)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
