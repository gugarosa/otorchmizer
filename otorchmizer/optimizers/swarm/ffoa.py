# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Fruit-Fly Optimization Algorithm.

References:
    W.-T. Pan.
    A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example.
    Knowledge-Based Systems (2012).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class FFOA(Optimizer):
    """Fruit-Fly Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize persistent x-axis and y-axis populations.

        Args:
            population: Population whose positions initialize both axes.

        """

        self.x_axis = population.positions.clone()
        self.y_axis = population.positions.clone()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one smell-based search step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        x = self.x_axis + torch.rand(
            pop.n_agents,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        y = self.y_axis + torch.rand(
            pop.n_agents,
            1,
            1,
            device=pop.device,
            dtype=pop.dtype,
        )
        distance = torch.sqrt(x.square() + y.square())
        smell_positions = distance.clamp_min(torch.finfo(pop.dtype).tiny).reciprocal()
        smell_positions = smell_positions.clamp(
            min=pop.lb.unsqueeze(0),
            max=pop.ub.unsqueeze(0),
        )
        smell_fitness = ctx.function(smell_positions)
        improved = smell_fitness < pop.fitness
        self.x_axis[improved] = x[improved]
        self.y_axis[improved] = y[improved]
        pop.positions[improved] = smell_positions[improved]
        pop.fitness[improved] = smell_fitness[improved]
        pop.update_best()
