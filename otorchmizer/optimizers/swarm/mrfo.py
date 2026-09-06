# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Manta Ray Foraging Optimization.

References:
    W. Zhao, Z. Zhang, and L. Wang.
    Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications.
    Engineering Applications of Artificial Intelligence (2020).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MRFO(Optimizer):
    """Manta Ray Foraging Optimization.

    Notes:
        Chain foraging, cyclone foraging, and somersault foraging phases.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> MRFO.")

        self.S = 2.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def S(self) -> float:
        """Return the somersault factor."""

        return self._S

    @S.setter
    def S(self, S: float) -> None:
        if not isinstance(S, (float, int)):
            raise e.TypeError("`S` must be a float or integer.")
        self._S = S

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        r1 = torch.rand(n, device=device, dtype=pop.dtype)

        for i in range(n):
            if r1[i].item() < 0.5:
                r = torch.rand(1, device=device, dtype=pop.dtype)
                beta = (
                    2
                    * torch.exp(r * (ctx.n_iterations - ctx.iteration + 1) / max(ctx.n_iterations, 1))
                    * torch.sin(2 * torch.pi * r)
                )

                if t < torch.rand(1, device=device, dtype=pop.dtype).item():
                    r_pos = torch.rand(pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * (
                        ub.squeeze(0) - lb.squeeze(0)
                    ) + lb.squeeze(0)
                    if i == 0:
                        new_pos = (
                            r_pos
                            + torch.rand(1, device=device, dtype=pop.dtype) * (r_pos - pop.positions[i])
                            + beta * (r_pos - pop.positions[i])
                        )
                    else:
                        new_pos = (
                            r_pos
                            + torch.rand(1, device=device, dtype=pop.dtype) * (pop.positions[i - 1] - pop.positions[i])
                            + beta * (r_pos - pop.positions[i])
                        )
                else:
                    if i == 0:
                        new_pos = (
                            best.squeeze(0)
                            + torch.rand(1, device=device, dtype=pop.dtype) * (best.squeeze(0) - pop.positions[i])
                            + beta * (best.squeeze(0) - pop.positions[i])
                        )
                    else:
                        new_pos = (
                            best.squeeze(0)
                            + torch.rand(1, device=device, dtype=pop.dtype) * (pop.positions[i - 1] - pop.positions[i])
                            + beta * (best.squeeze(0) - pop.positions[i])
                        )
            else:
                r = torch.rand(1, device=device, dtype=pop.dtype)
                alpha = 2 * r * torch.sqrt(torch.abs(torch.log(r + 1e-10)))

                if i == 0:
                    new_pos = (
                        pop.positions[i]
                        + r * (best.squeeze(0) - pop.positions[i])
                        + alpha * (best.squeeze(0) - pop.positions[i])
                    )
                else:
                    new_pos = (
                        pop.positions[i]
                        + r * (pop.positions[i - 1] - pop.positions[i])
                        + alpha * (best.squeeze(0) - pop.positions[i])
                    )

            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit

        r1 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        r2 = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        new_positions = pop.positions + self.S * (r1 * best - r2 * pop.positions)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
