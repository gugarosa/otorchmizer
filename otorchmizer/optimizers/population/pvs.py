"""Passing Vehicle Search.

References:
    P. Savsani and V. Savsani.
    Passing vehicle search (PVS): A novel metaheuristic algorithm.
    Applied Mathematical Modelling (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PVS(Optimizer):
    """Passing Vehicle Search.

    Vehicle overtaking dynamics.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> PVS.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        pop.positions = pop.positions[sorted_idx]
        pop.fitness = pop.fitness[sorted_idx]

        new_positions = pop.positions.clone()

        for i in range(n):
            # Select two random agents
            R = torch.randperm(n, device=device)
            R = R[R != i][:2]
            if len(R) < 2:
                continue
            r0, r1_idx = R[0], R[1]

            D1 = pop.fitness[i] / n
            D2 = pop.fitness[r0] / n
            D3 = pop.fitness[r1_idx] / n

            V1 = torch.rand(1, device=device) * (1 - D1)
            V2 = torch.rand(1, device=device) * (1 - D2)
            V3 = torch.rand(1, device=device) * (1 - D3)

            x = torch.abs(D3 - D1)
            y = torch.abs(D3 - D2)

            if V3 < V1:
                x1 = (V3 * x) / (V1 - V3 + 1e-10)
                y1 = (V2 * x) / (V1 - V3 + 1e-10)

                if (y - y1) > x1:
                    Vco = V1 / (V1 - V3 + 1e-10)
                    r = torch.rand(1, 1, device=device)
                    new_positions[i] = pop.positions[i] + Vco * r * (pop.positions[i] - pop.positions[r1_idx])
                else:
                    r = torch.rand(1, 1, device=device)
                    new_positions[i] = pop.positions[i] + r * (pop.positions[i] - pop.positions[r0])
            else:
                r = torch.rand(1, 1, device=device)
                new_positions[i] = pop.positions[i] + r * (pop.positions[r1_idx] - pop.positions[i])

        pop.positions = new_positions.clamp(min=lb, max=ub)
