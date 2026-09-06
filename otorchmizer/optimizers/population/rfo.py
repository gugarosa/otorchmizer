# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Red Fox Optimization.

References:
    D. Polap and M. Woźniak.
    Red fox optimization algorithm.
    Expert Systems with Applications (2021).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

import otorchmizer.math.general as g
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class RFO(Optimizer):
    """Apply fox relocation, noticing, and habitat replacement phases."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.phi = torch.rand(1).item() * math.tau
        self.theta = torch.rand(1).item()
        self.p_replacement = 0.05

        super().__init__(params)

    @property
    def phi(self) -> float:
        """Return the observation angle."""

        return self._phi

    @phi.setter
    def phi(self, phi: float) -> None:
        if not isinstance(phi, Real):
            raise TypeError("`phi` must be a float or integer.")
        if not 0 <= phi <= math.tau:
            raise ValueError("`phi` must be between 0 and 2π.")
        self._phi = float(phi)

    @property
    def theta(self) -> float:
        """Return the weather-condition coefficient."""

        return self._theta

    @theta.setter
    def theta(self, theta: float) -> None:
        if not isinstance(theta, Real):
            raise TypeError("`theta` must be a float or integer.")
        if not 0 <= theta <= 1:
            raise ValueError("`theta` must be between 0 and 1.")
        self._theta = float(theta)

    @property
    def p_replacement(self) -> float:
        """Return the habitat replacement ratio."""

        return self._p_replacement

    @p_replacement.setter
    def p_replacement(self, p_replacement: float) -> None:
        if not isinstance(p_replacement, Real):
            raise TypeError("`p_replacement` must be a float or integer.")
        if not 0 <= p_replacement <= 1:
            raise ValueError("`p_replacement` must be between 0 and 1.")
        self._p_replacement = float(p_replacement)

    def compile(self, population) -> None:
        """Calculate the number of foxes replaced per update.

        Args:
            population: Population whose size determines replacement count.

        """

        self.n_replacement = int(self.p_replacement * population.n_agents)

    def update(self, ctx: UpdateContext) -> None:
        """Relocate foxes and replace agents in the least-fit habitats.

        Args:
            ctx: Current optimization state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        alpha = torch.rand(1, device=device, dtype=pop.dtype) * 0.2

        sorted_idx = torch.argsort(pop.fitness)
        best_pos = pop.positions[sorted_idx[0]].clone()

        for i in range(n):
            dist = g.euclidean_distance(pop.positions[i].reshape(-1), best_pos.reshape(-1))
            dist = torch.sqrt(dist + 1e-10)
            sign = torch.sign(best_pos - pop.positions[i])
            candidate = pop.positions[i] + torch.rand(1, device=device, dtype=pop.dtype) * dist * sign
            candidate = candidate.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            candidate_fitness = ctx.function(candidate.unsqueeze(0))[0]
            if candidate_fitness < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fitness

            if torch.rand(1, device=device, dtype=pop.dtype).item() > 0.75:
                if self.phi != 0:
                    radius = alpha * torch.sin(torch.tensor(self.phi, device=device, dtype=pop.dtype)) / self.phi
                else:
                    radius = torch.tensor(self.theta, device=device, dtype=pop.dtype)

                angles = (
                    torch.rand(
                        pop.n_variables,
                        device=device,
                        dtype=pop.dtype,
                    )
                    * math.tau
                )
                sine = torch.sin(angles)
                prior_sine = torch.cumsum(sine, dim=0) - sine
                displacement = alpha * radius * (prior_sine + torch.cos(angles))
                candidate = (pop.positions[i] + displacement.unsqueeze(-1)).clamp(
                    min=lb.squeeze(0),
                    max=ub.squeeze(0),
                )
                candidate_fitness = ctx.function(candidate.unsqueeze(0))[0]
                if candidate_fitness < pop.fitness[i]:
                    pop.positions[i] = candidate
                    pop.fitness[i] = candidate_fitness

        pop.update_best()
        sorted_idx = torch.argsort(pop.fitness)
        best_pos = pop.positions[sorted_idx[0]].clone()
        second_pos = pop.positions[sorted_idx[1]].clone() if n > 1 else best_pos
        center = ((best_pos + second_pos) / 2).clone()
        diameter = g.euclidean_distance(best_pos.reshape(-1), second_pos.reshape(-1))
        diameter = torch.sqrt(diameter + 1e-10)

        worst_idx = sorted_idx.flip(0)[: self.n_replacement]
        if worst_idx.numel() > 0:
            replacement_positions = []
            for idx in worst_idx:
                k = torch.rand(1, device=device, dtype=pop.dtype)
                if k.item() >= 0.45:
                    random_position = pop.lb + torch.rand_like(pop.positions[idx]) * (pop.ub - pop.lb)
                    candidate = random_position + center + diameter / 2
                else:
                    candidate = k * center
                replacement_positions.append(candidate)

            replacements = torch.stack(replacement_positions).clamp(min=lb, max=ub)
            replacement_fitness = ctx.function(replacements)
            pop.positions[worst_idx] = replacements
            pop.fitness[worst_idx] = replacement_fitness

        pop.update_best()
