# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Boolean Manta Ray Foraging Optimization.

Updates combine cyclone or chain foraging with a final somersault phase.

References:
    Publication pending.

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BMRFO(Optimizer):
    """Boolean Manta Ray Foraging Optimization.

    Notes:
        Expresses chain, cyclone, and somersault foraging with Boolean operations.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the BMRFO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.S = 1.0
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one BMRFO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position
        t = ctx.iteration / max(ctx.n_iterations, 1)

        for i in range(n):
            r1 = torch.rand(1, device=device).item()
            pos = pop.positions[i].bool()

            if r1 < 0.5:
                # Cyclone foraging (binary)
                r_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                beta_bin = torch.round(torch.rand_like(pop.positions[i])).bool()

                if t < torch.rand(1, device=device).item():
                    r_pos = torch.round(torch.rand_like(pop.positions[i])).bool()
                    ref = pop.positions[max(i - 1, 0)].bool() if i > 0 else r_pos
                    p1 = r_bin | (ref ^ pos)
                    p2 = beta_bin | (r_pos ^ pos)
                    new_pos = r_pos & p1 & p2
                else:
                    best_b = best.bool()
                    ref = pop.positions[max(i - 1, 0)].bool() if i > 0 else best_b
                    p1 = r_bin | (ref ^ pos)
                    p2 = beta_bin | (best_b ^ pos)
                    new_pos = best_b & p1 & p2
            else:
                # Chain foraging (binary)
                r_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                alpha_bin = torch.round(torch.rand_like(pop.positions[i])).bool()
                best_b = best.bool()

                if i == 0:
                    p1 = r_bin & (best_b ^ pos)
                    p2 = alpha_bin & (best_b ^ pos)
                else:
                    prev = pop.positions[i - 1].bool()
                    p1 = r_bin & (prev ^ pos)
                    p2 = alpha_bin & (best_b ^ pos)
                new_pos = pos | p1 | p2

            pop.positions[i] = new_pos.to(dtype=pop.dtype)

        pop.fitness = fn(pop.positions)
        pop.update_best()

        # Somersault foraging
        best_b = pop.best_position.bool()
        S_b = torch.tensor(self.S, device=device).bool() if self.S else torch.zeros(1, device=device).bool()

        for i in range(n):
            pos = pop.positions[i].bool()
            r1 = torch.round(torch.rand_like(pop.positions[i])).bool()
            r2 = torch.round(torch.rand_like(pop.positions[i])).bool()
            somersault = pos | (S_b & ((r1 ^ best_b) ^ (r2 ^ pos)))
            pop.positions[i] = somersault.to(dtype=pop.dtype)

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        pop.positions = pop.positions.clamp(min=lb, max=ub)
