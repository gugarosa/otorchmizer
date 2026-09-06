# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Crow Search Algorithm.

References:
    A. Askarzadeh.
    A novel metaheuristic method for solving constrained engineering
    optimization problems: Crow search algorithm.
    Computers & Structures (2016).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class CSA(Optimizer):
    """Crow Search Algorithm.

    Notes:
        Vectorized memory-based crow search.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> CSA.")

        self.fl = 2.0
        self.AP = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def fl(self) -> float:
        """Return the flight length."""

        return self._fl

    @fl.setter
    def fl(self, fl: float) -> None:
        if not isinstance(fl, (float, int)):
            raise e.TypeError("`fl` must be a float or integer.")
        self._fl = fl

    @property
    def AP(self) -> float:
        """Return the awareness probability."""

        return self._AP

    @AP.setter
    def AP(self, AP: float) -> None:
        if not isinstance(AP, (float, int)):
            raise e.TypeError("`AP` must be a float or integer.")
        if not 0 <= AP <= 1:
            raise e.ValueError("`AP` must be between 0 and 1.")
        self._AP = AP

    def compile(self, population: Population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        self.memory = population.positions.clone()
        self.memory_fitness = torch.full_like(population.fitness, torch.inf)

    def evaluate(self, population: Population, function: Function) -> None:
        """Evaluate the population and synchronize each crow's memory.

        Args:
            population: Population to evaluate.
            function: Objective function applied to the population.

        """

        new_fitness = function(population.positions)
        improved = new_fitness < self.memory_fitness
        self.memory[improved] = population.positions[improved].clone()
        self.memory_fitness[improved] = new_fitness[improved].clone()
        population.fitness = new_fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        j = torch.randint(0, n, (n,), device=device)

        aware = torch.rand(n, device=device, dtype=pop.dtype)

        r_fl = torch.rand(n, 1, 1, device=device, dtype=pop.dtype) * self.fl
        toward_memory = pop.positions + r_fl * (self.memory[j] - pop.positions)

        random_pos = torch.rand(n, pop.n_variables, pop.n_dimensions, device=device, dtype=pop.dtype) * (ub - lb) + lb

        not_aware = (aware >= self.AP).view(n, 1, 1)
        new_positions = torch.where(not_aware, toward_memory, random_pos)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]

        mem_improved = new_fitness < self.memory_fitness
        self.memory[mem_improved] = new_positions[mem_improved]
        self.memory_fitness[mem_improved] = new_fitness[mem_improved]
