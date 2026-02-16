"""Manta Ray Foraging Optimization.

References:
    W. Zhao, Z. Zhang, and L. Wang.
    Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications.
    Engineering Applications of Artificial Intelligence (2020).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MRFO(Optimizer):
    """Manta Ray Foraging Optimization.

    Chain foraging, cyclone foraging, and somersault foraging phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> MRFO.")

        self.S = 2.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def S(self) -> float:
        return self._S

    @S.setter
    def S(self, S: float) -> None:
        if not isinstance(S, (float, int)):
            raise e.TypeError("`S` should be a float or integer")
        self._S = S

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        r1 = torch.rand(n, device=device)

        for i in range(n):
            if r1[i].item() < 0.5:
                # Cyclone foraging
                r = torch.rand(1, device=device)
                beta = 2 * torch.exp(r * (ctx.n_iterations - ctx.iteration + 1) / max(ctx.n_iterations, 1)) * torch.sin(2 * torch.pi * r)

                if t < torch.rand(1, device=device).item():
                    # Random reference
                    r_pos = torch.rand(pop.n_variables, pop.n_dimensions, device=device) * (ub.squeeze(0) - lb.squeeze(0)) + lb.squeeze(0)
                    if i == 0:
                        new_pos = r_pos + torch.rand(1, device=device) * (r_pos - pop.positions[i]) + beta * (r_pos - pop.positions[i])
                    else:
                        new_pos = r_pos + torch.rand(1, device=device) * (pop.positions[i - 1] - pop.positions[i]) + beta * (r_pos - pop.positions[i])
                else:
                    if i == 0:
                        new_pos = best.squeeze(0) + torch.rand(1, device=device) * (best.squeeze(0) - pop.positions[i]) + beta * (best.squeeze(0) - pop.positions[i])
                    else:
                        new_pos = best.squeeze(0) + torch.rand(1, device=device) * (pop.positions[i - 1] - pop.positions[i]) + beta * (best.squeeze(0) - pop.positions[i])
            else:
                # Chain foraging
                r = torch.rand(1, device=device)
                alpha = 2 * r * torch.sqrt(torch.abs(torch.log(r + 1e-10)))

                if i == 0:
                    new_pos = pop.positions[i] + r * (best.squeeze(0) - pop.positions[i]) + alpha * (best.squeeze(0) - pop.positions[i])
                else:
                    new_pos = pop.positions[i] + r * (pop.positions[i - 1] - pop.positions[i]) + alpha * (best.squeeze(0) - pop.positions[i])

            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit

        # Somersault foraging
        r1 = torch.rand(n, 1, 1, device=device)
        r2 = torch.rand(n, 1, 1, device=device)
        new_positions = pop.positions + self.S * (r1 * best - r2 * pop.positions)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
