# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Krill Herd.

References:
    A. H. Gandomi and A. H. Alavi.
    Krill herd: a new bio-inspired optimization algorithm.
    Communications in Nonlinear Science and Numerical Simulation (2012).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class KH(Optimizer):
    """Krill Herd optimizer.

    Notes:
        Induced motion, foraging activity, and physical diffusion.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> KH.")

        self.N_max = 0.01
        self.V_f = 0.02
        self.D_max = 0.002
        self.C_t = 0.93
        self.W_n = 0.42
        self.W_f = 0.38
        self.d_s = 2.63
        self.nn = 5
        self.Cr = 0.2
        self.Mu = 0.05

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def N_max(self) -> float:
        """Return the maximum induced speed."""

        return self._N_max

    @N_max.setter
    def N_max(self, N_max: float) -> None:
        if not isinstance(N_max, (float, int)):
            raise e.TypeError("`N_max` must be a float or integer.")
        self._N_max = N_max

    @property
    def V_f(self) -> float:
        """Return the foraging speed."""

        return self._V_f

    @V_f.setter
    def V_f(self, V_f: float) -> None:
        if not isinstance(V_f, (float, int)):
            raise e.TypeError("`V_f` must be a float or integer.")
        self._V_f = V_f

    @property
    def D_max(self) -> float:
        """Return the maximum diffusion speed."""

        return self._D_max

    @D_max.setter
    def D_max(self, D_max: float) -> None:
        if not isinstance(D_max, (float, int)):
            raise e.TypeError("`D_max` must be a float or integer.")
        self._D_max = D_max

    @property
    def C_t(self) -> float:
        """Return the position constant."""

        return self._C_t

    @C_t.setter
    def C_t(self, C_t: float) -> None:
        if not isinstance(C_t, (float, int)):
            raise e.TypeError("`C_t` must be a float or integer.")
        self._C_t = C_t

    def compile(self, population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.induced_motion = torch.zeros(shape, device=population.device, dtype=population.dtype)
        self.foraging_motion = torch.zeros(shape, device=population.device, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)

        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        K = (pop.fitness - best_fit) / (worst_fit - best_fit + c.EPSILON)

        flat = pop.positions.reshape(n, -1)
        dist_matrix = torch.cdist(flat, flat)

        _, nn_idx = dist_matrix.topk(min(self.nn + 1, n), largest=False, dim=1)
        nn_idx = nn_idx[:, 1:]

        diff = pop.positions.unsqueeze(1) - pop.positions[nn_idx]
        nn_dist = dist_matrix.gather(1, nn_idx).clamp(min=1e-10)
        K_diff = (K.unsqueeze(0).expand(n, -1).gather(1, nn_idx) - K.unsqueeze(1)) / (nn_dist + c.EPSILON)
        alpha_local = K_diff.unsqueeze(-1).unsqueeze(-1) * diff / nn_dist.unsqueeze(-1).unsqueeze(-1)
        alpha_local = alpha_local.sum(dim=1)

        dist_to_best = torch.linalg.norm((pop.positions - best).reshape(n, -1), dim=1).clamp(min=1e-10)
        K_best = (K - 0.0) / (dist_to_best + c.EPSILON)
        alpha_target = K_best.view(n, 1, 1) * (best - pop.positions) / dist_to_best.view(n, 1, 1)

        alpha = alpha_local + alpha_target
        self.induced_motion = self.W_n * self.induced_motion + self.N_max * alpha

        food_pos = (pop.positions / (pop.fitness.view(n, 1, 1) + c.EPSILON)).sum(dim=0) / (
            1.0 / (pop.fitness + c.EPSILON)
        ).sum()
        food_pos = food_pos.unsqueeze(0)
        dist_to_food = torch.linalg.norm((pop.positions - food_pos).reshape(n, -1), dim=1).clamp(min=1e-10)
        K_food = K / (dist_to_food + c.EPSILON)

        beta_food = K_food.view(n, 1, 1) * (food_pos - pop.positions) / dist_to_food.view(n, 1, 1)
        beta_best = K_best.view(n, 1, 1) * (best - pop.positions) / dist_to_best.view(n, 1, 1)
        beta = beta_food + beta_best

        self.foraging_motion = self.W_f * self.foraging_motion + self.V_f * beta

        diffusion = self.D_max * (1 - t) * (torch.rand_like(pop.positions) * 2 - 1)

        dt = self.C_t * ((ub - lb).sum() / pop.n_variables)
        pop.positions = pop.positions + dt * (self.induced_motion + self.foraging_motion + diffusion)
        pop.positions = pop.positions.clamp(min=lb, max=ub)

        Cr = self.Cr * (1 - t)
        cr_mask = torch.rand_like(pop.positions) < Cr
        rand_idx = torch.randint(0, n, (n,), device=device)
        pop.positions = torch.where(cr_mask, pop.positions[rand_idx], pop.positions)

        Mu = self.Mu / (t + c.EPSILON)
        mu_mask = torch.rand_like(pop.positions) < min(Mu, 1.0)
        rand_pos = torch.rand_like(pop.positions) * (ub - lb) + lb
        pop.positions = torch.where(mu_mask, rand_pos, pop.positions)
        pop.positions = pop.positions.clamp(min=lb, max=ub)
