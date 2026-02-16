"""Artificial Bee Colony.

References:
    D. Karaboga. An idea based on honey bee swarm for numerical optimization.
    Technical report, Erciyes University (2005).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ABC(Optimizer):
    """Artificial Bee Colony optimizer.

    Vectorized employed-bee, onlooker-bee, and scout-bee phases.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> ABC.")

        self.n_trials = 10

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def n_trials(self) -> int:
        return self._n_trials

    @n_trials.setter
    def n_trials(self, n_trials: int) -> None:
        if not isinstance(n_trials, int):
            raise e.TypeError("`n_trials` should be an integer")
        if n_trials <= 0:
            raise e.ValueError("`n_trials` should be > 0")
        self._n_trials = n_trials

    def compile(self, population) -> None:
        self.trial = torch.zeros(population.n_agents, device=population.device)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        device = pop.device

        # --- Employed Bee Phase ---
        k = torch.randint(0, n, (n,), device=device)
        # Ensure k != i
        arange = torch.arange(n, device=device)
        mask = k == arange
        k[mask] = (k[mask] + 1) % n

        j = torch.randint(0, pop.n_variables, (n,), device=device)
        phi = torch.rand(n, device=device) * 2 - 1  # [-1, 1]

        new_positions = pop.positions.clone()
        for i in range(n):
            new_positions[i, j[i], :] = (
                pop.positions[i, j[i], :]
                + phi[i] * (pop.positions[i, j[i], :] - pop.positions[k[i], j[i], :])
            )

        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)
        improved = new_fitness < pop.fitness
        pop.positions[improved] = new_positions[improved]
        pop.fitness[improved] = new_fitness[improved]
        self.trial[improved] = 0
        self.trial[~improved] += 1

        # --- Onlooker Bee Phase ---
        max_fit = pop.fitness.max()
        probs = (max_fit - pop.fitness + c.EPSILON) / (max_fit - pop.fitness.min() + c.EPSILON)
        probs = probs / probs.sum()

        selected = torch.multinomial(probs, n, replacement=True)

        k2 = torch.randint(0, n, (n,), device=device)
        j2 = torch.randint(0, pop.n_variables, (n,), device=device)
        phi2 = torch.rand(n, device=device) * 2 - 1

        new_positions2 = pop.positions[selected].clone()
        for i in range(n):
            si = selected[i]
            new_positions2[i, j2[i], :] = (
                pop.positions[si, j2[i], :]
                + phi2[i] * (pop.positions[si, j2[i], :] - pop.positions[k2[i], j2[i], :])
            )

        new_positions2 = new_positions2.clamp(min=lb, max=ub)
        new_fitness2 = fn(new_positions2)

        for i in range(n):
            si = selected[i]
            if new_fitness2[i] < pop.fitness[si]:
                pop.positions[si] = new_positions2[i]
                pop.fitness[si] = new_fitness2[i]
                self.trial[si] = 0
            else:
                self.trial[si] += 1

        # --- Scout Bee Phase ---
        scouts = self.trial >= self.n_trials
        if scouts.any():
            n_scouts = scouts.sum().item()
            pop.positions[scouts] = torch.rand(n_scouts, pop.n_variables, pop.n_dimensions, device=device) * (ub - lb) + lb
            pop.fitness[scouts] = fn(pop.positions[scouts])
            self.trial[scouts] = 0
