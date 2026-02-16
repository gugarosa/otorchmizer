"""Crow Search Algorithm.

References:
    A. Askarzadeh.
    A novel metaheuristic method for solving constrained engineering
    optimization problems: Crow search algorithm.
    Computers & Structures (2016).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class CSA(Optimizer):
    """Crow Search Algorithm.

    Vectorized memory-based crow search.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> CSA.")

        self.fl = 2.0
        self.AP = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def fl(self) -> float:
        return self._fl

    @fl.setter
    def fl(self, fl: float) -> None:
        if not isinstance(fl, (float, int)):
            raise e.TypeError("`fl` should be a float or integer")
        self._fl = fl

    @property
    def AP(self) -> float:
        return self._AP

    @AP.setter
    def AP(self, AP: float) -> None:
        if not isinstance(AP, (float, int)):
            raise e.TypeError("`AP` should be a float or integer")
        if not 0 <= AP <= 1:
            raise e.ValueError("`AP` should be between 0 and 1")
        self._AP = AP

    def compile(self, population) -> None:
        self.memory = population.positions.clone()
        self.memory_fitness = population.fitness.clone()

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Select random crow to follow
        j = torch.randint(0, n, (n,), device=device)

        # Awareness probability check
        r = torch.rand(n, 1, 1, device=device)
        aware = torch.rand(n, device=device)

        # If aware > AP: move toward memory of crow j
        # Else: random position
        r_fl = torch.rand(n, 1, 1, device=device) * self.fl
        toward_memory = pop.positions + r_fl * (self.memory[j] - pop.positions)

        random_pos = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb) + lb

        not_aware = (aware >= self.AP).view(n, 1, 1)
        new_positions = torch.where(not_aware, toward_memory, random_pos)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        # Update positions
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        # Update memory
        mem_improved = new_fitness < self.memory_fitness
        self.memory[mem_improved] = new_positions[mem_improved]
        self.memory_fitness[mem_improved] = new_fitness[mem_improved]
