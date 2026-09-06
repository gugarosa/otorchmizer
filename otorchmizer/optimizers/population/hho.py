# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Harris Hawks Optimization.

References:
    A. A. Heidari et al.
    Harris hawks optimization: Algorithm and applications.
    Future Generation Computer Systems (2019).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.distribution as d
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class HHO(Optimizer):
    """Apply exploration, besiege, and rapid-dive Harris hawk strategies."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> HHO.")

        super().__init__(params)

        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        """Move each hawk according to its sampled energy and strategy.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        avg = pop.positions.mean(dim=0, keepdim=True)

        for i in range(n):
            E0 = 2 * torch.rand(1, device=device, dtype=pop.dtype) - 1
            J = 2 * (1 - torch.rand(1, device=device, dtype=pop.dtype))
            E = 2 * E0 * (1 - t)

            if E.abs().item() >= 1:
                q = torch.rand(1, device=device, dtype=pop.dtype)
                j = torch.randint(0, n, (1,), device=device).item()
                if q.item() >= 0.5:
                    r = torch.rand(4, device=device, dtype=pop.dtype)
                    new_pos = pop.positions[j] - r[0] * torch.abs(pop.positions[j] - 2 * r[1] * pop.positions[i])
                else:
                    r = torch.rand(4, device=device, dtype=pop.dtype)
                    new_pos = (best.squeeze(0) - avg.squeeze(0)) - r[2] * (
                        lb.squeeze(0) + r[3] * (ub.squeeze(0) - lb.squeeze(0))
                    )
            else:
                w = torch.rand(1, device=device, dtype=pop.dtype)
                delta = best.squeeze(0) - pop.positions[i]

                if w.item() >= 0.5:
                    if E.abs().item() >= 0.5:
                        new_pos = delta - E.abs() * torch.abs(J * best.squeeze(0) - pop.positions[i])
                    else:
                        new_pos = best.squeeze(0) - E.abs() * delta.abs()
                else:
                    if E.abs().item() >= 0.5:
                        Y = best.squeeze(0) - E.abs() * torch.abs(J * best.squeeze(0) - pop.positions[i])
                        S = torch.rand_like(Y)
                        levy = d.generate_levy_distribution(beta=1.5, size=Y.shape, device=device, dtype=pop.dtype)
                        Z = Y + S * levy

                        Y_fit = fn(Y.clamp(min=lb.squeeze(0), max=ub.squeeze(0)).unsqueeze(0))[0]
                        Z_fit = fn(Z.clamp(min=lb.squeeze(0), max=ub.squeeze(0)).unsqueeze(0))[0]

                        if Y_fit < pop.fitness[i]:
                            new_pos = Y
                        elif Z_fit < pop.fitness[i]:
                            new_pos = Z
                        else:
                            new_pos = pop.positions[i]
                    else:
                        Y = best.squeeze(0) - E.abs() * torch.abs(J * best.squeeze(0) - avg.squeeze(0))
                        S = torch.rand_like(Y)
                        levy = d.generate_levy_distribution(beta=1.5, size=Y.shape, device=device, dtype=pop.dtype)
                        Z = Y + S * levy

                        Y_fit = fn(Y.clamp(min=lb.squeeze(0), max=ub.squeeze(0)).unsqueeze(0))[0]
                        Z_fit = fn(Z.clamp(min=lb.squeeze(0), max=ub.squeeze(0)).unsqueeze(0))[0]

                        if Y_fit < pop.fitness[i]:
                            new_pos = Y
                        elif Z_fit < pop.fitness[i]:
                            new_pos = Z
                        else:
                            new_pos = pop.positions[i]

            pop.positions[i] = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
