"""Water Cycle Algorithm.

References:
    H. Eskandar et al.
    Water cycle algorithm – A novel metaheuristic optimization method
    for solving constrained engineering optimization problems.
    Computers & Structures (2012).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class WCA(Optimizer):
    """Water Cycle Algorithm.

    Sea, rivers, and streams flow dynamics.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> WCA.")

        self.nsr = 2
        self.d_max = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def nsr(self) -> int:
        return self._nsr

    @nsr.setter
    def nsr(self, nsr: int) -> None:
        if not isinstance(nsr, int):
            raise e.TypeError("`nsr` should be an integer")
        if nsr <= 1:
            raise e.ValueError("`nsr` should be > 1")
        self._nsr = nsr

    @property
    def d_max(self) -> float:
        return self._d_max

    @d_max.setter
    def d_max(self, d_max: float) -> None:
        if not isinstance(d_max, (float, int)):
            raise e.TypeError("`d_max` should be a float or integer")
        self._d_max = d_max

    def compile(self, population) -> None:
        n = population.n_agents
        # Flow intensity: how many streams each river/sea gets
        self.flows = torch.zeros(self.nsr, dtype=torch.long, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
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
        cost = pop.fitness[:self.nsr].abs()
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
            pop.positions[stream_start:end] = pop.positions[stream_start:end] + r * (pop.positions[i].unsqueeze(0) - pop.positions[stream_start:end])
            stream_start = end

        # Update rivers toward sea
        for i in range(1, self.nsr):
            r = torch.rand(1, 1, device=device)
            pop.positions[i] = pop.positions[i] + r * (pop.positions[0] - pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
        pop.fitness = fn(pop.positions)

        # Raining: if river too close to sea
        for i in range(1, self.nsr):
            dist = torch.linalg.norm((pop.positions[i] - pop.positions[0]).reshape(-1))
            if dist < self.d_max:
                pop.positions[i] = torch.rand(pop.n_variables, pop.n_dimensions, device=device) * (ub.squeeze(0) - lb.squeeze(0)) + lb.squeeze(0)
                pop.fitness[i] = fn(pop.positions[i].unsqueeze(0))[0]

        # Decay d_max
        self.d_max *= 0.99

        pop.update_best()
