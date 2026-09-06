# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Algorithm of the Innovative Gunner.

References:
    P. Pijarski and P. Kacejko.
    A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
    Engineering Optimization (2019).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class AIG(Optimizer):
    """Algorithm of the Innovative Gunner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the AIG optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = torch.pi
        self.beta = torch.pi
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one AIG step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        angle_scale = torch.rand((), device=device, dtype=pop.dtype)
        alpha_max = self.alpha * angle_scale
        beta_max = self.beta * angle_scale

        for i in range(n):
            alpha_corr = torch.randn_like(pop.positions[i]) * alpha_max / 3
            beta_corr = torch.randn_like(pop.positions[i]) * beta_max / 3
            alpha_cosine = torch.cos(alpha_corr)
            beta_cosine = torch.cos(beta_corr)
            alpha_reciprocal = alpha_cosine.reciprocal()
            beta_reciprocal = beta_cosine.reciprocal()
            if ((alpha_corr >= 0) & ~torch.isfinite(alpha_reciprocal)).any() or (
                (beta_corr >= 0) & ~torch.isfinite(beta_reciprocal)
            ).any():
                raise ValueError("`angular correction` produced an unrepresentable reciprocal.")
            g_alpha = torch.where(alpha_corr < 0, alpha_cosine, alpha_reciprocal)
            g_beta = torch.where(beta_corr < 0, beta_cosine, beta_reciprocal)

            new_pos = pop.positions[i] * g_alpha * g_beta
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit

        pop.update_best()
