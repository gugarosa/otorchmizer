"""Social Spider Optimization.

References:
    E. Cuevas et al.
    A swarm optimization algorithm inspired in the behavior of the social-spider.
    Expert Systems with Applications (2013).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SSO(Optimizer):
    """Social Spider Optimization.

    Gender-based movement with mating and female/male dynamics.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> SSO.")

        self.female_percentage = 0.65
        self.PF = 0.7

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def female_percentage(self) -> float:
        return self._female_percentage

    @female_percentage.setter
    def female_percentage(self, female_percentage: float) -> None:
        if not isinstance(female_percentage, (float, int)):
            raise e.TypeError("`female_percentage` should be a float or integer")
        if not 0 <= female_percentage <= 1:
            raise e.ValueError("`female_percentage` should be between 0 and 1")
        self._female_percentage = female_percentage

    @property
    def PF(self) -> float:
        return self._PF

    @PF.setter
    def PF(self, PF: float) -> None:
        if not isinstance(PF, (float, int)):
            raise e.TypeError("`PF` should be a float or integer")
        if not 0 <= PF <= 1:
            raise e.ValueError("`PF` should be between 0 and 1")
        self._PF = PF

    def compile(self, population) -> None:
        n = population.n_agents
        self.n_female = max(int(n * self.female_percentage), 1)
        self.n_male = n - self.n_female
        self.weight = torch.zeros(n, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        # Calculate weights based on fitness
        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        self.weight = (worst_fit - pop.fitness) / (worst_fit - best_fit + 1e-10)

        # --- Female movement ---
        for i in range(self.n_female):
            r = torch.rand(1, device=device)
            alpha = torch.rand(1, 1, device=device)
            beta = torch.rand(1, 1, device=device)
            delta = torch.rand(1, 1, device=device)

            # Select nearest high-weight spider
            j = torch.randint(0, n, (1,), device=device).item()

            if r < self.PF:
                pop.positions[i] = (
                    pop.positions[i]
                    + alpha * self.weight[j] * (pop.positions[j] - pop.positions[i])
                    + beta * self.weight[0] * (best.squeeze(0) - pop.positions[i])
                    + delta * (torch.rand_like(pop.positions[i]) - 0.5)
                )
            else:
                pop.positions[i] = (
                    pop.positions[i]
                    - alpha * self.weight[j] * (pop.positions[j] - pop.positions[i])
                    - beta * self.weight[0] * (best.squeeze(0) - pop.positions[i])
                    + delta * (torch.rand_like(pop.positions[i]) - 0.5)
                )

        # --- Male movement ---
        if self.n_male > 0:
            male_start = self.n_female
            male_weights = self.weight[male_start:]
            median_weight = male_weights.median()

            for i in range(male_start, n):
                alpha = torch.rand(1, 1, device=device)
                delta = torch.rand(1, 1, device=device)

                if self.weight[i] > median_weight:
                    # Dominant male
                    pop.positions[i] = (
                        pop.positions[i]
                        + alpha * (best.squeeze(0) - pop.positions[i])
                        + delta * (torch.rand_like(pop.positions[i]) - 0.5)
                    )
                else:
                    # Non-dominant male: weighted mean position of females
                    female_weights = self.weight[:self.n_female].view(-1, 1, 1)
                    female_mean = (female_weights * pop.positions[:self.n_female]).sum(dim=0) / female_weights.sum()
                    pop.positions[i] = pop.positions[i] + alpha * (female_mean - pop.positions[i])

        pop.positions = pop.positions.clamp(min=lb, max=ub)
