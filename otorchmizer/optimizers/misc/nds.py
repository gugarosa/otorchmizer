# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Non-dominated sorting of precomputed objective vectors."""

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class NDS(Optimizer):
    """Assign Pareto-front ranks without moving or scalarizing objective vectors.

    Notes:
        Positions contain precomputed objective values, normally supplied through ParetoSpace.
        Maximization matches the migration source and can be disabled with maximize=False.
        Status zero is the nondominated front, with larger values for subsequent fronts.
        The public count and set tensors store domination counts and the quadratic-size domination matrix.
        Fitness stores front ranks, and best_position is a current first-front representative, not a historical optimum.

    References:
        P. Godfrey, R. Shipley, and J. Gryz.
        Algorithms and Analyses for Maximal Vector Computation.
        The VLDB Journal (2007).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize nondominated sorting.

        Args:
            params: Algorithm overrides, including maximize for objective orientation.

        Raises:
            TypeError: The maximize flag is not a Boolean.

        """

        self.maximize = True
        self.n_pareto_points = 0
        super().__init__(params)
        if not isinstance(self.maximize, bool):
            raise e.TypeError("`maximize` must be a Boolean.")

    def compile(self, population: Population) -> None:
        """Allocate front ranks and domination state.

        Args:
            population: Population containing objective vectors.

        """

        n = population.n_agents
        self.count = torch.zeros(n, dtype=torch.long, device=population.device)
        self.set = torch.zeros((n, n), dtype=torch.bool, device=population.device)
        self.status = torch.full((n,), -1, dtype=torch.long, device=population.device)
        self.n_pareto_points = 0

    def evaluate(self, population: Population, function: Function | None = None) -> None:
        """Rank the stored objective vectors without invoking the scalar objective.

        Args:
            population: Population containing objective vectors.
            function: Unused scalar objective accepted by the common optimizer interface.

        Raises:
            ValueError: An objective value is NaN.

        """

        values = population.positions.flatten(1)
        if torch.isnan(values).any():
            raise e.ValueError("`population.positions` must not contain NaN objective values.")

        self.set.fill_(True)
        strict = torch.zeros_like(self.set)
        for objective in values.unbind(dim=1):
            left = objective.unsqueeze(1)
            right = objective.unsqueeze(0)
            self.set &= left >= right if self.maximize else left <= right
            strict |= left > right if self.maximize else left < right
        self.set &= strict
        self.count = self.set.sum(dim=0)
        self.status.fill_(-1)

        remaining = self.count.clone()
        front = remaining == 0
        self.n_pareto_points = int(front.sum().item())
        rank = 0
        while front.any():
            self.status[front] = rank
            remaining -= self.set[front].sum(dim=0)
            front = (remaining == 0) & (self.status < 0)
            rank += 1

        population.fitness = self.status.to(dtype=population.dtype)
        first = self.status.argmin()
        population.best_fitness = population.fitness[first].clone()
        population.best_position = population.positions[first].clone()

    def update(self, ctx: UpdateContext) -> None:
        """Refresh the front ranks without changing objective vectors.

        Args:
            ctx: Update context containing the objective-vector population.

        """

        self.evaluate(ctx.space.population, ctx.function)
