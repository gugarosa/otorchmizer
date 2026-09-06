# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Cohort Intelligence.

References:
    A. J. Kulkarni, I. P. Durugkar, M. Kumar. Cohort Intelligence: A Self Supervised Learning Behavior.
    IEEE International Conference on Systems, Man, and Cybernetics (2013).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


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
            raise ValueError("`population.fitness` must not contain NaN or negative infinity.")

        scored = torch.isfinite(fitness)
        if not scored.any():
            raise ValueError("`population.fitness` must contain at least one finite value.")

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
