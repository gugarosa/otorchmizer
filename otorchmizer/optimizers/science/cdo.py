# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Chernobyl Disaster Optimizer.

References:
    H. A. Shehadeh. Chernobyl disaster optimizer (CDO): a novel meta-heuristic method for global optimization.
    Neural Computing and Applications (2023). https://doi.org/10.1007/s00521-023-08261-1

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


def _nonzero(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


class CDO(Optimizer):
    """Chernobyl Disaster Optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the CDO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if population.n_agents < 3:
            raise ValueError("`population.n_agents` must be at least 3 for CDO.")

        shape = (population.n_variables, population.n_dimensions)
        self.gamma_pos = population.positions.new_zeros(shape)
        self.beta_pos = population.positions.new_zeros(shape)
        self.alpha_pos = population.positions.new_zeros(shape)
        self.gamma_fit = population.fitness.new_full((), torch.inf)
        self.beta_fit = population.fitness.new_full((), torch.inf)
        self.alpha_fit = population.fitness.new_full((), torch.inf)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one CDO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        leaders = torch.argsort(pop.fitness)[:3]
        self.alpha_pos, self.alpha_fit = pop.positions[leaders[0]].clone(), pop.fitness[leaders[0]].clone()
        self.beta_pos, self.beta_fit = pop.positions[leaders[1]].clone(), pop.fitness[leaders[1]].clone()
        self.gamma_pos, self.gamma_fit = pop.positions[leaders[2]].clone(), pop.fitness[leaders[2]].clone()

        ws = 3 * (1 - t)
        sampling_dtype = torch.float64 if pop.dtype == torch.float64 else torch.float32
        one = torch.tensor(1.0, device=pop.device, dtype=sampling_dtype)
        s_gamma = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 299999).to(pop.dtype)
        s_beta = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 269999).to(pop.dtype)
        s_alpha = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 15999).to(pop.dtype)
        s_gamma, s_beta, s_alpha = _nonzero(s_gamma), _nonzero(s_beta), _nonzero(s_alpha)

        def component(target, source_scale, denominator):
            r1 = torch.rand_like(pop.positions)
            r2 = torch.rand_like(pop.positions)
            r3 = torch.rand_like(pop.positions)
            rho = torch.pi * r1.square() / denominator - ws * r2
            gradient = (torch.pi * r3.square() * target.unsqueeze(0) - pop.positions).abs()
            return source_scale * (pop.positions - rho * gradient)

        v_gamma = component(self.gamma_pos, 1.0, s_gamma)
        v_beta = component(self.beta_pos, 0.5, 0.5 * s_beta)
        v_alpha = component(self.alpha_pos, 0.25, 0.25 * s_alpha)
        pop.positions = ((v_alpha + v_beta + v_gamma) / 3).clamp(min=lb, max=ub)
