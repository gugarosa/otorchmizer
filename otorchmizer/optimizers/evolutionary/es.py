"""Evolution Strategies.

References:
    H.-G. Beyer and H.-P. Schwefel.
    Evolution strategies – A comprehensive introduction.
    Natural Computing (2002).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ES(Optimizer):
    """Evolution Strategies (μ, λ)-ES.

    Self-adaptive mutation strategy with (μ+λ) selection.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> ES.")

        self.child_ratio = 0.5

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def child_ratio(self) -> float:
        return self._child_ratio

    @child_ratio.setter
    def child_ratio(self, child_ratio: float) -> None:
        if not isinstance(child_ratio, (float, int)):
            raise e.TypeError("`child_ratio` should be a float or integer")
        if not 0 <= child_ratio <= 1:
            raise e.ValueError("`child_ratio` should be between 0 and 1")
        self._child_ratio = child_ratio

    def compile(self, population) -> None:
        n = population.n_agents
        self.n_children = max(int(n * self.child_ratio), 1)
        shape = (n, population.n_variables, population.n_dimensions)
        lb = population.lb.unsqueeze(0)
        ub = population.ub.unsqueeze(0)
        self.strategy = torch.zeros(shape, device=population.device)
        # Initialize strategy for first n_children parents
        self.strategy[:self.n_children] = 0.05 * torch.rand(
            self.n_children, population.n_variables, population.n_dimensions,
            device=population.device
        ) * (ub - lb)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        nc = self.n_children

        # Self-adaptive strategy update
        tau = 1.0 / (2.0 * pop.n_variables) ** 0.5
        tau_p = 1.0 / (2.0 * (pop.n_variables ** 0.5)) ** 0.5

        new_strategy = self.strategy[:nc].clone()
        r1 = torch.randn(nc, 1, 1, device=device)
        r2 = torch.randn(nc, pop.n_variables, pop.n_dimensions, device=device)
        new_strategy = new_strategy * torch.exp(tau_p * r1 + tau * r2)

        # Mutate parents to create children
        children = pop.positions[:nc] + new_strategy * torch.randn(nc, pop.n_variables, pop.n_dimensions, device=device)
        children = children.clamp(min=lb, max=ub)
        children_fitness = fn(children)

        # Combine and select best n
        all_pos = torch.cat([pop.positions, children], dim=0)
        all_fit = torch.cat([pop.fitness, children_fitness], dim=0)
        all_strategy = torch.cat([self.strategy, new_strategy], dim=0)

        sorted_idx = torch.argsort(all_fit)[:n]
        pop.positions = all_pos[sorted_idx]
        pop.fitness = all_fit[sorted_idx]
        self.strategy = all_strategy[sorted_idx]
