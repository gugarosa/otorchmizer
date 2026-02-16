"""Remaining science-based optimizers: AIG, CDO, EFO, ESA, HGSO, LSA, MOA, SMA, TEO, TWO, WEO."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.distribution as d
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class AIG(Optimizer):
    """Algorithm of the Innovative Gunner."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.alpha = 3.14159
        self.beta = 3.14159
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        alpha_max = self.alpha * (1 - t)
        beta_max = self.beta * (1 - t)

        for i in range(n):
            alpha_corr = torch.randn(1, device=device) * alpha_max
            beta_corr = torch.randn(1, device=device) * beta_max

            g_alpha = torch.cos(alpha_corr) if alpha_corr < 0 else 1.0 / torch.cos(alpha_corr).clamp(min=0.01)
            g_beta = torch.cos(beta_corr) if beta_corr < 0 else 1.0 / torch.cos(beta_corr).clamp(min=0.01)

            new_pos = pop.positions[i] * g_alpha * g_beta
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit


class CDO(Optimizer):
    """Chernobyl Disaster Optimizer."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)

    def compile(self, population) -> None:
        shape = (population.n_variables, population.n_dimensions)
        device = population.device
        self.gamma_pos = torch.zeros(shape, device=device)
        self.beta_pos = torch.zeros(shape, device=device)
        self.alpha_pos = torch.zeros(shape, device=device)
        self.gamma_fit = torch.tensor(c.FLOAT_MAX, device=device)
        self.beta_fit = torch.tensor(c.FLOAT_MAX, device=device)
        self.alpha_fit = torch.tensor(c.FLOAT_MAX, device=device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Update top-3
        for i in range(n):
            f = pop.fitness[i]
            if f < self.alpha_fit:
                self.gamma_pos, self.gamma_fit = self.beta_pos.clone(), self.beta_fit.clone()
                self.beta_pos, self.beta_fit = self.alpha_pos.clone(), self.alpha_fit.clone()
                self.alpha_pos, self.alpha_fit = pop.positions[i].clone(), f.clone()
            elif f < self.beta_fit:
                self.gamma_pos, self.gamma_fit = self.beta_pos.clone(), self.beta_fit.clone()
                self.beta_pos, self.beta_fit = pop.positions[i].clone(), f.clone()
            elif f < self.gamma_fit:
                self.gamma_pos, self.gamma_fit = pop.positions[i].clone(), f.clone()

        ws = 3 - 3 * t

        for i in range(n):
            r = torch.rand(3, device=device)
            s1 = torch.log(r[0] + 1e-10)
            s2 = torch.log(r[1] + 1e-10)
            s3 = torch.log(r[2] + 1e-10)

            v1 = self.alpha_pos + ws * torch.randn_like(self.alpha_pos) * s1
            v2 = self.beta_pos + ws * torch.randn_like(self.beta_pos) * s2
            v3 = self.gamma_pos + ws * torch.randn_like(self.gamma_pos) * s3

            pop.positions[i] = (v1 + v2 + v3) / 3

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class EFO(Optimizer):
    """Electromagnetic Field Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.positive_field = 0.1
        self.negative_field = 0.5
        self.ps_ratio = 0.1
        self.r_ratio = 0.4
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        sorted_idx = torch.argsort(pop.fitness)
        phi = (1 + 5**0.5) / 2

        n_pos = max(int(n * self.positive_field), 1)
        n_neg = max(int(n * self.negative_field), 1)

        for i in range(n):
            pos_i = torch.randint(0, n_pos, (1,), device=device).item()
            neg_i = torch.randint(n - n_neg, n, (1,), device=device).item()
            neu_i = torch.randint(n_pos, n - n_neg, (1,), device=device).item() if n > n_pos + n_neg else pos_i

            new_pos = pop.positions[i].clone()
            for j in range(pop.n_variables):
                r = torch.rand(1, device=device).item()
                if r < self.ps_ratio:
                    new_pos[j] = pop.positions[sorted_idx[pos_i], j]
                else:
                    pos_val = pop.positions[sorted_idx[pos_i], j]
                    neg_val = pop.positions[sorted_idx[neg_i], j]
                    neu_val = pop.positions[sorted_idx[neu_i], j]
                    force = torch.rand(1, device=device)
                    new_pos[j] = neu_val + phi * force * (pos_val - neu_val) + force * (neg_val - neu_val)

                if torch.rand(1, device=device).item() < self.r_ratio:
                    new_pos[j] = torch.rand(pop.n_dimensions, device=device) * (ub.squeeze(0)[j] - lb.squeeze(0)[j]) + lb.squeeze(0)[j]

            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            worst_idx = sorted_idx[-1]
            if new_fit < pop.fitness[worst_idx]:
                pop.positions[worst_idx] = new_pos
                pop.fitness[worst_idx] = new_fit


class ESA(Optimizer):
    """Electro-Search Algorithm."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.n_electrons = 5
        super().__init__(params)

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.D = torch.rand(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        for i in range(n):
            best_electron_pos = pop.positions[i].clone()
            best_electron_fit = pop.fitness[i].clone()

            for _ in range(self.n_electrons):
                n_level = torch.randint(2, 7, (1,), device=device).item()
                r = torch.rand_like(pop.positions[i]) * 2 - 1
                electron_pos = pop.positions[i] + r * (1 - 1.0 / n_level ** 2) / self.D[i]
                electron_pos = electron_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
                electron_fit = fn(electron_pos.unsqueeze(0))[0]

                if electron_fit < best_electron_fit:
                    best_electron_pos = electron_pos
                    best_electron_fit = electron_fit

            if best_electron_fit < pop.fitness[i]:
                self.D[i] = torch.abs(best_electron_pos - pop.positions[i]) + 1e-10
                pop.positions[i] = best_electron_pos
                pop.fitness[i] = best_electron_fit


class HGSO(Optimizer):
    """Henry Gas Solubility Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.n_clusters = 2
        self.alpha = 1.0
        self.beta = 1.0
        self.K = 1.0
        self.l1 = 0.0005
        self.l2 = 100.0
        self.l3 = 0.001
        super().__init__(params)

    def compile(self, population) -> None:
        n = population.n_agents
        device = population.device
        self.coeff = torch.full((n,), self.l1, device=device)
        self.pressure = torch.full((n,), self.l2, device=device)
        self.constant = torch.full((n,), self.l3, device=device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        T_val = torch.exp(torch.tensor(-t, device=device))
        self.coeff = self.coeff * torch.exp(-self.constant / T_val)

        sorted_idx = torch.argsort(pop.fitness)

        for i in range(n):
            solubility = self.K * self.coeff[i] * self.pressure[i]
            gamma = self.beta * torch.exp(-(pop.best_fitness + 0.05) / (pop.fitness[i] + 0.05))
            flag = 1 if torch.rand(1, device=device).item() > 0.5 else -1

            r = torch.rand(1, 1, device=device)
            new_pos = pop.positions[i] + flag * r * gamma * (best.squeeze(0) - pop.positions[i]) + self.alpha * r * solubility * (pop.positions[sorted_idx[i % n]] - pop.positions[i])
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit

        # Replace worst 10-20%
        n_replace = max(int(n * 0.1), 1)
        worst_idx = torch.argsort(pop.fitness, descending=True)[:n_replace]
        pop.positions[worst_idx] = torch.rand(n_replace, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb) + lb
        pop.fitness[worst_idx] = fn(pop.positions[worst_idx])


class LSA(Optimizer):
    """Lightning Search Algorithm."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.max_time = 10
        self.E = 2.05
        self.p_fork = 0.01
        super().__init__(params)

    def compile(self, population) -> None:
        self.time = 0
        n = population.n_agents
        self.direction = torch.sign(torch.randn(n, population.n_variables, population.n_dimensions, device=population.device))

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        self.time += 1
        if self.time >= self.max_time:
            worst_idx = pop.fitness.argmax()
            pop.positions[worst_idx] = pop.best_position.clone()
            pop.fitness[worst_idx] = pop.best_fitness.clone()
            self.time = 0

        energy = self.E - 2 * torch.exp(torch.tensor(-5 * (1 - t), device=device))

        for i in range(n):
            r = torch.rand(1, device=device)
            noise = torch.randn_like(pop.positions[i])

            if r.item() < 0.5:
                pop.positions[i] = pop.positions[i] + energy * self.direction[i] * torch.abs(noise) * (best.squeeze(0) - pop.positions[i])
            else:
                pop.positions[i] = best.squeeze(0) + energy * noise

            if torch.rand(1, device=device).item() < self.p_fork:
                j = torch.randint(0, pop.n_variables, (1,), device=device).item()
                pop.positions[i, j] = torch.rand(pop.n_dimensions, device=device) * (ub.squeeze(0)[j] - lb.squeeze(0)[j]) + lb.squeeze(0)[j]

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class MOA(Optimizer):
    """Magnetic Optimization Algorithm."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.alpha = 1.0
        self.rho = 2.0
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        sorted_idx = torch.argsort(pop.fitness)
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()

        norm_fit = (pop.fitness - best_fit) / (worst_fit - best_fit + c.EPSILON)
        mass = self.alpha + self.rho * norm_fit

        for i in range(n):
            force = torch.zeros_like(pop.positions[i])
            neighbors = [max(i - 1, 0), min(i + 1, n - 1)]

            for j in neighbors:
                if j == i:
                    continue
                diff = pop.positions[j] - pop.positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                force += norm_fit[j] * diff / dist

            vel = force / (mass[i] + 1e-10) * torch.rand(1, device=device)
            pop.positions[i] = pop.positions[i] + vel


class SMA(Optimizer):
    """Slime Mould Algorithm."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.z = 0.03
        super().__init__(params)

    def compile(self, population) -> None:
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.weight = torch.ones(shape, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]
        fit_range = worst_fit - best_fit + c.EPSILON

        # Update weights
        for rank, idx in enumerate(sorted_idx):
            r = torch.rand(pop.n_variables, pop.n_dimensions, device=device)
            log_val = torch.log10((best_fit - pop.fitness[idx]) / fit_range + 1).abs()
            if rank < n // 2:
                self.weight[idx] = 1 + r * log_val
            else:
                self.weight[idx] = 1 - r * log_val

        a_val = torch.atanh(torch.tensor(-(t + 1) / (max(ctx.n_iterations, 1) + 1) + 1, device=device)).clamp(max=5)
        b_val = 1 - (t + 1) / (max(ctx.n_iterations, 1) + 1)

        for i in range(n):
            r = torch.rand(1, device=device).item()
            if r < self.z:
                pop.positions[i] = torch.rand_like(pop.positions[i]) * (ub.squeeze(0) - lb.squeeze(0)) + lb.squeeze(0)
            else:
                p = torch.tanh(torch.abs(pop.fitness[i] - best_fit))
                vb = torch.rand_like(pop.positions[i]) * 2 * a_val - a_val
                vc = torch.rand_like(pop.positions[i]) * 2 * b_val - b_val

                if torch.rand(1, device=device).item() < p.item():
                    k = torch.randint(0, n, (1,), device=device).item()
                    l_idx = torch.randint(0, n, (1,), device=device).item()
                    pop.positions[i] = best.squeeze(0) + vb * self.weight[i] * (pop.positions[k] - pop.positions[l_idx])
                else:
                    pop.positions[i] = pop.positions[i] * vc

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class TEO(Optimizer):
    """Thermal Exchange Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.c1 = 1.0
        self.c2 = 1.0
        self.pro = 0.05
        self.n_TM = 4
        super().__init__(params)

    def compile(self, population) -> None:
        self.environment = population.positions.clone()
        self.TM = []

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        # Thermal memory
        self.TM.append(pop.best_position.clone())
        if len(self.TM) > self.n_TM:
            self.TM = self.TM[-self.n_TM:]

        # Replace worst with TM members
        sorted_idx = torch.argsort(pop.fitness, descending=True)
        for k, tm_pos in enumerate(self.TM):
            if k < n:
                pop.positions[sorted_idx[k]] = tm_pos
                pop.fitness[sorted_idx[k]] = fn(tm_pos.unsqueeze(0))[0]

        # Update environment
        r = torch.rand(n, 1, 1, device=device)
        self.environment = (1 - (self.c1 + self.c2 * (1 - t)) * r) * self.environment

        # Update positions
        worst_fit = pop.fitness.max()
        for i in range(n):
            beta = pop.fitness[i] / (worst_fit + 1e-10)
            pop.positions[i] = pop.positions[i] + (self.environment[i] - pop.positions[i]) * torch.exp(torch.tensor(-beta * t, device=device))

            if torch.rand(1, device=device).item() < self.pro:
                j = torch.randint(0, pop.n_variables, (1,), device=device).item()
                pop.positions[i, j] = torch.rand(pop.n_dimensions, device=device) * (ub.squeeze(0)[j] - lb.squeeze(0)[j]) + lb.squeeze(0)[j]

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class TWO(Optimizer):
    """Tug of War Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.mu_s = 1.0
        self.mu_k = 1.0
        self.delta_t = 1.0
        self.alpha_val = 0.9
        self.beta_val = 0.05
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration + 1

        sorted_idx = torch.argsort(pop.fitness)
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        weights = (pop.fitness - worst_fit) / (best_fit - worst_fit + c.EPSILON) + 1

        new_positions = pop.positions.clone()

        for i in range(n):
            delta = torch.zeros_like(pop.positions[i])
            for j in range(n):
                if i == j or weights[i] >= weights[j]:
                    continue
                diff = pop.positions[j] - pop.positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                force = (weights[j] - weights[i]) / dist
                accel = force / (weights[i] + 1e-10)
                r = torch.rand(1, device=device)
                delta += 0.5 * accel * self.delta_t ** 2 + self.alpha_val ** t * self.beta_val * (ub.squeeze(0) - lb.squeeze(0)) * r

            new_positions[i] = pop.positions[i] + delta

        # Constraint handling
        for i in range(n):
            out = (new_positions[i] < lb.squeeze(0)) | (new_positions[i] > ub.squeeze(0))
            if out.any():
                r = torch.rand_like(new_positions[i])
                correction = best.squeeze(0) + r / t * (best.squeeze(0) - new_positions[i])
                new_positions[i] = torch.where(out, correction, new_positions[i])

        pop.positions = new_positions.clamp(min=lb, max=ub)


class WEO(Optimizer):
    """Water Evaporation Optimization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.E_min = -3.5
        self.E_max = -0.5
        self.theta_min = -torch.pi / 3.6
        self.theta_max = -torch.pi / 9
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        for i in range(n):
            new_pos = pop.positions[i].clone()

            if t <= 0.5:
                # Monolayer evaporation
                E_sub = self.E_max - (self.E_max - self.E_min) * t * 2
                r = torch.rand_like(pop.positions[i])
                MEP = (r < torch.exp(torch.tensor(E_sub, device=device))).float()
            else:
                # Droplet evaporation
                theta = self.theta_max - (self.theta_max - self.theta_min) * (t - 0.5) * 2
                cos_t = torch.cos(torch.tensor(theta, device=device))
                J = (1 / 2.6) * ((2 / 3 + cos_t ** 3 / 3 - cos_t).clamp(min=1e-10) ** (-2 / 3)) * (1 - cos_t)
                r = torch.rand_like(pop.positions[i])
                MEP = (r < J).float()

            j = torch.randint(0, n, (1,), device=device).item()
            S = torch.rand(1, device=device) * (pop.positions[i] - pop.positions[j])

            new_pos = pop.positions[i] + S * MEP
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]

            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit
