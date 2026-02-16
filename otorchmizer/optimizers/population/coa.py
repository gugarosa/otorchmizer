"""Coyote Optimization Algorithm.

References:
    J. Pierezan and L. dos Santos Coelho.
    Coyote Optimization Algorithm: A new metaheuristic for global optimization problems.
    IEEE Congress on Evolutionary Computation (2018).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class COA(Optimizer):
    """Coyote Optimization Algorithm.

    Pack-based cultural tendency with alpha leadership.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> COA.")

        self.n_p = 2

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def n_p(self) -> int:
        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, int):
            raise e.TypeError("`n_p` should be an integer")
        if n_p <= 0:
            raise e.ValueError("`n_p` should be > 0")
        self._n_p = n_p

    def compile(self, population) -> None:
        self.n_c = population.n_agents // self.n_p

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for pack_i in range(self.n_p):
            start = pack_i * self.n_c
            end = start + self.n_c if pack_i < self.n_p - 1 else n

            pack_pos = pop.positions[start:end]
            pack_fit = pop.fitness[start:end]
            pack_n = pack_pos.shape[0]

            # Sort pack by fitness
            sorted_idx = torch.argsort(pack_fit)
            pack_pos = pack_pos[sorted_idx]
            pack_fit = pack_fit[sorted_idx]

            alpha = pack_pos[0].unsqueeze(0)

            # Cultural tendency (median)
            tendency = pack_pos.median(dim=0).values.unsqueeze(0)

            for j in range(pack_n):
                cr1 = torch.randint(0, pack_n, (1,), device=device).item()
                cr2 = torch.randint(0, pack_n, (1,), device=device).item()

                r1 = torch.rand(1, 1, device=device)
                r2 = torch.rand(1, 1, device=device)

                lambda1 = alpha.squeeze(0) - pack_pos[cr1]
                lambda2 = tendency.squeeze(0) - pack_pos[cr2]

                new_pos = pack_pos[j] + r1 * lambda1 + r2 * lambda2
                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]

                if new_fit < pack_fit[j]:
                    pack_pos[j] = new_pos
                    pack_fit[j] = new_fit

            pop.positions[start:end] = pack_pos[torch.argsort(sorted_idx)]
            pop.fitness[start:end] = pack_fit[torch.argsort(sorted_idx)]

        # Pack transition
        p_e = 0.005 * n
        if torch.rand(1, device=device).item() < p_e:
            i = torch.randint(0, n, (1,), device=device).item()
            j = torch.randint(0, n, (1,), device=device).item()
            pop.positions[[i, j]] = pop.positions[[j, i]]
            pop.fitness[[i, j]] = pop.fitness[[j, i]]
