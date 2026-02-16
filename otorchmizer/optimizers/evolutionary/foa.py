"""Forest Optimization Algorithm.

References:
    M. Ghaemi and M.-R. Feizi-Derakhshi.
    Forest Optimization Algorithm.
    Expert Systems with Applications (2014).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class FOA(Optimizer):
    """Forest Optimization Algorithm.

    Local seeding, population limiting, and global seeding phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
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
        return self._life_time

    @life_time.setter
    def life_time(self, life_time: int) -> None:
        if not isinstance(life_time, int):
            raise e.TypeError("`life_time` should be an integer")
        if life_time <= 0:
            raise e.ValueError("`life_time` should be > 0")
        self._life_time = life_time

    @property
    def area_limit(self) -> int:
        return self._area_limit

    @area_limit.setter
    def area_limit(self, area_limit: int) -> None:
        if not isinstance(area_limit, int):
            raise e.TypeError("`area_limit` should be an integer")
        if area_limit <= 0:
            raise e.ValueError("`area_limit` should be > 0")
        self._area_limit = area_limit

    @property
    def LSC(self) -> int:
        return self._LSC

    @LSC.setter
    def LSC(self, LSC: int) -> None:
        if not isinstance(LSC, int):
            raise e.TypeError("`LSC` should be an integer")
        if LSC <= 0:
            raise e.ValueError("`LSC` should be > 0")
        self._LSC = LSC

    @property
    def GSC(self) -> int:
        return self._GSC

    @GSC.setter
    def GSC(self, GSC: int) -> None:
        if not isinstance(GSC, int):
            raise e.TypeError("`GSC` should be an integer")
        if GSC <= 0:
            raise e.ValueError("`GSC` should be > 0")
        self._GSC = GSC

    @property
    def transfer_rate(self) -> float:
        return self._transfer_rate

    @transfer_rate.setter
    def transfer_rate(self, transfer_rate: float) -> None:
        if not isinstance(transfer_rate, (float, int)):
            raise e.TypeError("`transfer_rate` should be a float or integer")
        if not 0 <= transfer_rate <= 1:
            raise e.ValueError("`transfer_rate` should be between 0 and 1")
        self._transfer_rate = transfer_rate

    def compile(self, population) -> None:
        self.age = torch.zeros(population.n_agents, dtype=torch.long, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Local seeding: create offspring from zero-aged trees
        young_mask = self.age == 0
        young_idx = young_mask.nonzero(as_tuple=True)[0]

        new_positions_list = []
        new_fitness_list = []

        for idx in young_idx:
            for _ in range(self.LSC):
                offspring = pop.positions[idx].clone()
                j = torch.randint(0, pop.n_variables, (1,), device=device).item()
                offspring[j] = torch.rand(pop.n_dimensions, device=device) * (ub.squeeze(0)[j] - lb.squeeze(0)[j]) + lb.squeeze(0)[j]
                new_positions_list.append(offspring)

        if new_positions_list:
            new_pos = torch.stack(new_positions_list)
            new_fit = fn(new_pos)
            # Merge with population
            all_pos = torch.cat([pop.positions, new_pos], dim=0)
            all_fit = torch.cat([pop.fitness, new_fit], dim=0)
            all_age = torch.cat([self.age, torch.zeros(new_pos.shape[0], dtype=torch.long, device=device)])
        else:
            all_pos = pop.positions
            all_fit = pop.fitness
            all_age = self.age

        # Increment age
        all_age = all_age + 1

        # Remove trees older than life_time
        alive = all_age <= self.life_time
        all_pos = all_pos[alive]
        all_fit = all_fit[alive]
        all_age = all_age[alive]

        # Sort by fitness, keep best area_limit
        sorted_idx = torch.argsort(all_fit)
        keep = min(max(self.area_limit, n), all_pos.shape[0])
        all_pos = all_pos[sorted_idx[:keep]]
        all_fit = all_fit[sorted_idx[:keep]]
        all_age = all_age[sorted_idx[:keep]]

        # Reset best tree age
        all_age[0] = 0

        # Ensure population size stays at n
        if all_pos.shape[0] < n:
            deficit = n - all_pos.shape[0]
            new_random = torch.rand(deficit, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb) + lb
            new_random_fit = fn(new_random)
            all_pos = torch.cat([all_pos, new_random], dim=0)
            all_fit = torch.cat([all_fit, new_random_fit], dim=0)
            all_age = torch.cat([all_age, torch.zeros(deficit, dtype=torch.long, device=device)])
        elif all_pos.shape[0] > n:
            all_pos = all_pos[:n]
            all_fit = all_fit[:n]
            all_age = all_age[:n]

        pop.positions = all_pos
        pop.fitness = all_fit
        self.age = all_age
