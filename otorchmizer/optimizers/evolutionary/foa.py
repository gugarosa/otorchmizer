# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Forest Optimization Algorithm.

References:
    M. Ghaemi and M.-R. Feizi-Derakhshi.
    Forest Optimization Algorithm.
    Expert Systems with Applications (2014).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FOA(Optimizer):
    """Apply local seeding and population limiting with the Forest Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        logger.info("Overriding class: Optimizer -> FOA.")

        self.life_time = 6
        self.area_limit = 30
        self.LSC = 1
        self.GSC = 1
        self.transfer_rate = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def life_time(self) -> int:
        """Return the maximum tree age."""

        return self._life_time

    @life_time.setter
    def life_time(self, life_time: int) -> None:
        if not isinstance(life_time, int):
            raise e.TypeError("`life_time` must be an integer.")
        if life_time <= 0:
            raise e.ValueError("`life_time` must be positive.")
        self._life_time = life_time

    @property
    def area_limit(self) -> int:
        """Return the forest area limit."""

        return self._area_limit

    @area_limit.setter
    def area_limit(self, area_limit: int) -> None:
        if not isinstance(area_limit, int):
            raise e.TypeError("`area_limit` must be an integer.")
        if area_limit <= 0:
            raise e.ValueError("`area_limit` must be positive.")
        self._area_limit = area_limit

    @property
    def LSC(self) -> int:
        """Return the number of local seeding changes."""

        return self._LSC

    @LSC.setter
    def LSC(self, LSC: int) -> None:
        if not isinstance(LSC, int):
            raise e.TypeError("`LSC` must be an integer.")
        if LSC <= 0:
            raise e.ValueError("`LSC` must be positive.")
        self._LSC = LSC

    @property
    def GSC(self) -> int:
        """Return the number of global seeding changes."""

        return self._GSC

    @GSC.setter
    def GSC(self, GSC: int) -> None:
        if not isinstance(GSC, int):
            raise e.TypeError("`GSC` must be an integer.")
        if GSC <= 0:
            raise e.ValueError("`GSC` must be positive.")
        self._GSC = GSC

    @property
    def transfer_rate(self) -> float:
        """Return the global-seeding transfer ratio."""

        return self._transfer_rate

    @transfer_rate.setter
    def transfer_rate(self, transfer_rate: float) -> None:
        if not isinstance(transfer_rate, (float, int)):
            raise e.TypeError("`transfer_rate` must be a float or integer.")
        if not 0 <= transfer_rate <= 1:
            raise e.ValueError("`transfer_rate` must be between 0 and 1.")
        self._transfer_rate = transfer_rate

    def compile(self, population) -> None:
        """Initialize tree ages.

        Args:
            population: Population that defines the age-vector length and device.

        """

        self.age = torch.zeros(population.n_agents, dtype=torch.long, device=population.device)

    def _local_seeding(self, pop, function) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        young_idx = (self.age == 0).nonzero(as_tuple=True)[0]
        parent_age = self.age + 1
        if young_idx.numel() == 0:
            return pop.positions, pop.fitness, parent_age

        parent_idx = young_idx.repeat_interleave(self.LSC)
        children = pop.positions[parent_idx].clone()
        variables = torch.randint(
            0,
            pop.n_variables,
            (children.shape[0],),
            device=pop.device,
        )
        rows = torch.arange(children.shape[0], device=pop.device)
        child_lb = pop.lb[variables]
        child_ub = pop.ub[variables]
        children[rows, variables] += torch.rand_like(child_lb) * (child_ub - child_lb) + child_lb
        children = children.clamp(min=pop.lb.unsqueeze(0), max=pop.ub.unsqueeze(0))
        children_fitness = function(children)

        positions = torch.cat((pop.positions, children))
        fitness = torch.cat((pop.fitness, children_fitness))
        age = torch.cat(
            (
                parent_age,
                torch.zeros(children.shape[0], dtype=torch.long, device=pop.device),
            )
        )
        return positions, fitness, age

    def _limit_population(
        self,
        positions: torch.Tensor,
        fitness: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alive = age < self.life_time
        candidate_positions = positions[~alive]
        positions = positions[alive]
        fitness = fitness[alive]
        age = age[alive]

        sorted_idx = torch.argsort(fitness)
        positions = positions[sorted_idx]
        fitness = fitness[sorted_idx]
        age = age[sorted_idx]

        if positions.shape[0] > self.area_limit:
            candidate_positions = torch.cat((candidate_positions, positions[self.area_limit :]))
            positions = positions[: self.area_limit]
            fitness = fitness[: self.area_limit]
            age = age[: self.area_limit]

        return positions, fitness, age, candidate_positions

    def _global_seeding(self, pop, function, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_global = int(candidates.shape[0] * self.transfer_rate)
        if n_global == 0:
            return (
                pop.positions.new_empty((0, pop.n_variables, pop.n_dimensions)),
                pop.fitness.new_empty((0,)),
            )

        seeds = candidates[:n_global].clone()
        rows = torch.arange(n_global, device=pop.device)
        for _ in range(self.GSC):
            variables = torch.randint(0, pop.n_variables, (n_global,), device=pop.device)
            seed_lb = pop.lb[variables]
            seed_ub = pop.ub[variables]
            seeds[rows, variables] = torch.rand_like(seed_lb) * (seed_ub - seed_lb) + seed_lb

        seeds_fitness = function(seeds)
        return seeds, seeds_fitness

    def update(self, ctx: UpdateContext) -> None:
        """Run local seeding, population limiting, and global seeding.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function

        positions, fitness, age = self._local_seeding(pop, fn)
        positions, fitness, age, candidates = self._limit_population(positions, fitness, age)
        global_seeds, global_fitness = self._global_seeding(pop, fn, candidates)

        positions = torch.cat((positions, global_seeds))
        fitness = torch.cat((fitness, global_fitness))
        age = torch.cat(
            (
                age,
                torch.zeros(global_seeds.shape[0], dtype=torch.long, device=pop.device),
            )
        )
        sorted_idx = torch.argsort(fitness)
        pop.positions = positions[sorted_idx]
        pop.fitness = fitness[sorted_idx]
        self.age = age[sorted_idx]
        self.age[0] = 0
        pop.n_agents = pop.positions.shape[0]
        pop.update_best()
