# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Water Cycle Algorithm.

References:
    H. Eskandar et al.
    Water cycle algorithm – A novel metaheuristic optimization method
    for solving constrained engineering optimization problems.
    Computers & Structures (2012).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WCA(Optimizer):
    """Water Cycle Algorithm.

    Notes:
        Models candidate movement through sea, river, and stream flow dynamics.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the WCA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> WCA.")

        self.nsr = 2
        self.d_max = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def nsr(self) -> int:
        """Return the number of rivers plus the sea.

        Returns:
            int: Current number of rivers plus the sea.

        """

        return self._nsr

    @nsr.setter
    def nsr(self, nsr: int) -> None:
        """Set the number of rivers plus the sea.

        Args:
            nsr: New value for the number of rivers plus the sea.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(nsr, int):
            raise e.TypeError("`nsr` must be an integer.")
        if nsr <= 1:
            raise e.ValueError("`nsr` must be greater than 1.")
        self._nsr = nsr

    @property
    def d_max(self) -> float:
        """Return the maximum evaporation distance.

        Returns:
            float: Current maximum evaporation distance.

        """

        return self._d_max

    @d_max.setter
    def d_max(self, d_max: float) -> None:
        """Set the maximum evaporation distance.

        Args:
            d_max: New value for the maximum evaporation distance.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(d_max, (float, int)):
            raise e.TypeError("`d_max` must be a float or integer.")
        self._d_max = d_max

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if population.n_agents < self.nsr:
            raise e.SizeError("`population.n_agents` must be at least `nsr`.")

        self.flows = torch.zeros(self.nsr, dtype=torch.long, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one WCA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Sort: best is sea (index 0), next nsr-1 are rivers
        sorted_idx = torch.argsort(pop.fitness)
        pop.positions = pop.positions[sorted_idx]
        pop.fitness = pop.fitness[sorted_idx]

        # Flow intensity
        cost = pop.fitness[: self.nsr].abs()
        total_cost = cost.sum() + 1e-10
        n_streams = n - self.nsr
        self.flows = torch.floor(cost / total_cost * n_streams).long()
        # Adjust for rounding
        diff = n_streams - self.flows.sum().item()
        if diff > 0:
            self.flows[0] += diff

        # Update streams toward rivers/sea
        stream_start = self.nsr
        for i in range(self.nsr):
            n_flow = self.flows[i].item()
            if n_flow <= 0:
                continue
            end = min(stream_start + n_flow, n)

            r = torch.rand(end - stream_start, 1, 1, device=device)
            pop.positions[stream_start:end] = pop.positions[stream_start:end] + 2 * r * (
                pop.positions[i].unsqueeze(0) - pop.positions[stream_start:end]
            )
            stream_start = end

        # Update rivers toward sea
        for i in range(1, self.nsr):
            r = torch.rand(1, 1, device=device)
            pop.positions[i] = pop.positions[i] + 2 * r * (pop.positions[0] - pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        pop.fitness = fn(pop.positions)

        for i in range(1, self.nsr):
            for j in range(self.nsr, n):
                if pop.fitness[j] < pop.fitness[i]:
                    pop.positions[[i, j]] = pop.positions[[j, i]]
                    pop.fitness[[i, j]] = pop.fitness[[j, i]]

        for i in range(1, self.nsr):
            if pop.fitness[i] < pop.fitness[0]:
                pop.positions[[0, i]] = pop.positions[[i, 0]]
                pop.fitness[[0, i]] = pop.fitness[[i, 0]]

        pop.update_best()

        stream_start = self.nsr
        for i in range(self.nsr):
            end = min(stream_start + self.flows[i].item(), n)
            for j in range(stream_start, end):
                distance = torch.linalg.norm((pop.best_position - pop.positions[j]).reshape(-1))
                if distance < self.d_max:
                    if i == 0:
                        pop.positions[j] = pop.best_position + (0.1**0.5) * torch.randn_like(pop.positions[j])
                    else:
                        pop.positions[j] = torch.rand_like(pop.positions[j]) * (
                            ub.squeeze(0) - lb.squeeze(0)
                        ) + lb.squeeze(0)
                    pop.positions[j] = pop.positions[j].clamp(min=pop.lb, max=pop.ub)
                    pop.fitness[j] = fn(pop.positions[j].unsqueeze(0))[0]
            stream_start = end

        self.d_max -= self.d_max / max(ctx.n_iterations, 1)

        pop.update_best()
