# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Social-based optimizers: BSO, CI, ISA, MVPA, QSA, SSD."""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.math.general as g
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


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


class CI(Optimizer):
    """Cohort Intelligence."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the CI optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.r = 0.8
        self.t = 3
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.lower = population.lb.unsqueeze(0).expand_as(population.positions).clone()
        self.upper = population.ub.unsqueeze(0).expand_as(population.positions).clone()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one CI step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents

        # Weighted wheel selection
        fitness = pop.fitness.clone()
        if torch.isnan(fitness).any() or torch.isneginf(fitness).any():
            raise e.ValueError("`population.fitness` must not contain NaN or negative infinity.")

        scored = torch.isfinite(fitness)
        if not scored.any():
            raise e.ValueError("`population.fitness` must contain at least one finite value.")

        weights = torch.zeros_like(fitness)
        if (fitness[scored] > 0).all():
            weights[scored] = fitness[scored].min() / fitness[scored]
        else:
            eps = torch.finfo(fitness.dtype).eps
            scaled = fitness[scored] / fitness[scored].abs().max().clamp_min(eps)
            shifted = scaled - scaled.min()
            weights[scored] = (shifted + eps).reciprocal()

        weights = weights / weights.max()
        weights = weights / weights.sum()

        for i in range(n):
            s = torch.multinomial(weights, 1).item()

            width = (self.upper[i] - self.lower[i]) * self.r / 2
            self.lower[i] = pop.positions[s] - width
            self.upper[i] = pop.positions[s] + width
            self.lower[i] = self.lower[i].clamp(min=pop.lb)
            self.upper[i] = self.upper[i].clamp(max=pop.ub)

            for _ in range(self.t):
                new_pos = torch.rand_like(pop.positions[i]) * (self.upper[i] - self.lower[i]) + self.lower[i]
                new_pos = new_pos.clamp(min=pop.lb, max=pop.ub)
                new_fit = fn(new_pos.unsqueeze(0))[0]
                if new_fit < pop.fitness[i]:
                    pop.positions[i] = new_pos
                    pop.fitness[i] = new_fit


class ISA(Optimizer):
    """Interactive Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ISA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.w = 0.7
        self.tau = 0.3
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        n = population.n_agents
        if n < 2:
            raise e.SizeError("`population.n_agents` must be at least 2 for ISA.")

        shape = (n, population.n_variables, population.n_dimensions)
        self.local_position = population.positions.new_zeros(shape)
        self.velocity = population.positions.new_zeros(shape)
        self.local_fitness = population.fitness.new_full((n,), torch.inf)

    def evaluate(self, population, function) -> None:
        """Evaluate a population and update optimizer-specific best state.

        Args:
            population: Population whose tensors define the optimizer state.
            function: Objective function used to score the population.

        """

        fitness = function(population.positions)
        improved = fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved]
        self.local_fitness[improved] = fitness[improved]
        population.fitness = fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ISA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Weighted position
        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]

        coef = (best_fit - pop.fitness) / (best_fit - worst_fit + c.EPSILON)
        w_coef = coef / (coef.sum() + c.EPSILON)
        w_position = (w_coef.view(n, 1, 1) * pop.positions).sum(dim=0)

        for i in range(n):
            r1 = torch.rand(1, device=device).item()
            idx = torch.randint(0, n, (1,), device=device).item()
            while idx == i:
                idx = torch.randint(0, n, (1,), device=device).item()

            if r1 >= self.tau:
                phi3 = torch.rand(1, device=device)
                phi2 = 2 * torch.rand(1, device=device)
                phi1 = -(phi2 + phi3) * torch.rand(1, device=device)

                self.velocity[i] = (
                    self.w * self.velocity[i]
                    + phi1 * (self.local_position[idx] - pop.positions[i])
                    + phi2 * (best.squeeze(0) - self.local_position[idx])
                    + phi3 * (w_position - self.local_position[idx])
                )
            else:
                r2 = torch.rand(1, 1, device=device)
                if pop.fitness[i] < pop.fitness[idx]:
                    self.velocity[i] = r2 * (pop.positions[i] - pop.positions[idx])
                else:
                    self.velocity[i] = r2 * (pop.positions[idx] - pop.positions[i])

            pop.positions[i] = pop.positions[i] + self.velocity[i]

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class MVPA(Optimizer):
    """Most Valuable Player Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the MVPA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_teams = 4
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if self.n_teams < 2:
            raise e.ValueError("`n_teams` must be at least 2.")
        if population.n_agents < self.n_teams:
            raise e.SizeError("`population.n_agents` must be at least `n_teams`.")

        self.n_p = population.n_agents // self.n_teams

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one MVPA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for team_i in range(self.n_teams):
            start_i = team_i * self.n_p
            end_i = start_i + self.n_p if team_i < self.n_teams - 1 else n

            team_pos = pop.positions[start_i:end_i]
            team_fit = pop.fitness[start_i:end_i]
            team_n = team_pos.shape[0]

            sorted_ti = torch.argsort(team_fit)
            franchise_i = team_pos[sorted_ti[0]].clone()
            fitness_i = team_fit.mean()

            # Select random opponent team
            j = torch.randint(0, self.n_teams, (1,), device=device).item()
            while j == team_i:
                j = torch.randint(0, self.n_teams, (1,), device=device).item()

            start_j = j * self.n_p
            end_j = start_j + self.n_p if j < self.n_teams - 1 else n
            team_j_fit = pop.fitness[start_j:end_j]
            team_j_pos = pop.positions[start_j:end_j]
            sorted_tj = torch.argsort(team_j_fit)
            franchise_j = team_j_pos[sorted_tj[0]].clone()
            fitness_j = team_j_fit.mean()

            for k in range(team_n):
                idx = start_i + k
                r1 = torch.rand(1, 1, device=device)
                r2 = torch.rand(1, device=device)
                r3 = torch.rand(1, 1, device=device)

                new_pos = (
                    pop.positions[idx]
                    + r1 * (franchise_i - pop.positions[idx])
                    + 2 * r1 * (best.squeeze(0) - pop.positions[idx])
                )

                Pr = 1 - fitness_i / (fitness_i + fitness_j + c.EPSILON)
                if r2.item() < Pr:
                    new_pos = new_pos + r3 * (pop.positions[idx] - franchise_j)
                else:
                    new_pos = new_pos + r3 * (franchise_j - pop.positions[idx])

                new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                new_fit = fn(new_pos.unsqueeze(0))[0]
                if new_fit < pop.fitness[idx]:
                    pop.positions[idx] = new_pos
                    pop.fitness[idx] = new_fit


class QSA(Optimizer):
    """Queuing Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the QSA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population) -> None:
        """Validate the population required by the three queue leaders.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            SizeError: If fewer than three agents are available.

        """

        if population.n_agents < 3:
            raise e.SizeError("`population.n_agents` must be at least 3 for QSA.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one QSA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration + 1
        T = max(ctx.n_iterations, 1)

        import math

        beta = math.exp(math.log(1 / (t + c.EPSILON)) * math.sqrt(t / T))

        sorted_idx = torch.argsort(pop.fitness)
        pop.positions = pop.positions[sorted_idx]
        pop.fitness = pop.fitness[sorted_idx]

        for i in range(n):
            alpha = torch.rand(1, device=device) * 2 - 1
            from otorchmizer.math.random import generate_gamma_random_number

            E = generate_gamma_random_number(1.0, 0.5, (pop.n_variables, pop.n_dimensions), device)
            e = generate_gamma_random_number(1.0, 0.5, (1,), device)

            # Business 1: move toward top-3
            idx_target = 0 if i < n // 3 else (1 if i < 2 * n // 3 else min(2, n - 1))
            A = pop.positions[idx_target]

            new_pos = A + beta * alpha * E * torch.abs(A - pop.positions[i]) + e * (A - pop.positions[i])
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit


class SSD(Optimizer):
    """Social Ski Driver."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SSD optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c_val = 2.0
        self.decay = 0.99
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        n = population.n_agents
        shape = (n, population.n_variables, population.n_dimensions)
        self.local_position = population.positions.new_zeros(shape)
        self.velocity = torch.rand_like(population.positions)
        self.local_fitness = population.fitness.new_full((n,), torch.inf)

    def evaluate(self, population, function) -> None:
        """Evaluate a population and update optimizer-specific best state.

        Args:
            population: Population whose tensors define the optimizer state.
            function: Objective function used to score the population.

        """

        fitness = function(population.positions)
        improved = fitness < self.local_fitness
        self.local_position[improved] = population.positions[improved]
        self.local_fitness[improved] = fitness[improved]
        population.fitness = fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one SSD step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        alpha_pos = pop.positions[sorted_idx[0]]
        beta_pos = pop.positions[sorted_idx[1]] if n > 1 else alpha_pos
        gamma_pos = pop.positions[sorted_idx[2]] if n > 2 else beta_pos

        mean = (alpha_pos + beta_pos + gamma_pos) / 3

        for i in range(n):
            r1 = torch.rand(1, device=device)
            r2 = torch.rand(1, device=device)

            # Update position
            pop.positions[i] = pop.positions[i] + self.velocity[i]

            # Update velocity
            if r2.item() <= 0.5:
                self.velocity[i] = self.c_val * torch.sin(r1) * (self.local_position[i] - pop.positions[i]) + torch.sin(
                    r1
                ) * (mean - pop.positions[i])
            else:
                self.velocity[i] = self.c_val * torch.cos(r1) * (self.local_position[i] - pop.positions[i]) + torch.cos(
                    r1
                ) * (mean - pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        self.c_val *= self.decay
