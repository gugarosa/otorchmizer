# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Brain Storm Optimization.

References:
    Y. Shi. Brain Storm Optimization Algorithm.
    International Conference in Swarm Intelligence (2011).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.general as g
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class BSO(Optimizer):
    """Brain Storm Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the BSO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.m = 5
        self.p_replacement_cluster = 0.2
        self.p_single_cluster = 0.8
        self.p_single_best = 0.4
        self.p_double_best = 0.5
        self.k = 20.0
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one BSO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        # K-means clustering
        labels = g.kmeans_torch(pop.positions, n_clusters=min(self.m, n))
        cluster_best = []
        cluster_members = []

        for ci in range(self.m):
            mask = labels == ci
            members = mask.nonzero(as_tuple=True)[0]
            cluster_members.append(members)
            if len(members) > 0:
                best_in_cluster = members[pop.fitness[members].argmin()]
                cluster_best.append(best_in_cluster.item())
            else:
                cluster_best.append(-1)

        # Replacement
        if torch.rand(1, device=device).item() < self.p_replacement_cluster:
            ci = torch.randint(0, self.m, (1,), device=device).item()
            if cluster_best[ci] >= 0:
                pop.positions[cluster_best[ci]] = torch.rand(pop.n_variables, pop.n_dimensions, device=device) * (
                    ub.squeeze(0) - lb.squeeze(0)
                ) + lb.squeeze(0)

        for i in range(n):
            new_pos = pop.positions[i].clone()

            if torch.rand(1, device=device).item() < self.p_single_cluster:
                ci = torch.randint(0, self.m, (1,), device=device).item()
                members = cluster_members[ci]
                if len(members) > 0:
                    if torch.rand(1, device=device).item() < self.p_single_best:
                        new_pos = pop.positions[cluster_best[ci]].clone()
                    else:
                        j = members[torch.randint(0, len(members), (1,), device=device).item()]
                        new_pos = pop.positions[j].clone()
            else:
                if self.m > 1:
                    c1, c2 = torch.randperm(self.m, device=device)[:2].tolist()
                    m1, m2 = cluster_members[c1], cluster_members[c2]
                    if len(m1) > 0 and len(m2) > 0:
                        if torch.rand(1, device=device).item() < self.p_double_best:
                            new_pos = (pop.positions[cluster_best[c1]] + pop.positions[cluster_best[c2]]) / 2
                        else:
                            u = m1[torch.randint(0, len(m1), (1,), device=device).item()]
                            v = m2[torch.randint(0, len(m2), (1,), device=device).item()]
                            new_pos = (pop.positions[u] + pop.positions[v]) / 2

            r = torch.rand(1, device=device)
            csi = torch.sigmoid((0.5 * T - t) / torch.tensor(self.k, device=device)) * r
            new_pos = new_pos + csi * torch.randn_like(new_pos)
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))

            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit
