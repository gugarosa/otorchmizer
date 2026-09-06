# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Differential Evolution.

References:
    R. Storn and K. Price.
    Differential Evolution – A Simple and Efficient Heuristic for
    Global Optimization over Continuous Spaces.
    Journal of Global Optimization (1997).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


def _sample_excluding(exclusions: torch.Tensor, n: int) -> torch.Tensor:
    sample = torch.randint(0, n - exclusions.shape[1], (exclusions.shape[0],), device=exclusions.device)
    for excluded in exclusions.sort(dim=1).values.unbind(dim=1):
        sample += sample >= excluded
    return sample


class DE(Optimizer):
    """Apply vectorized Differential Evolution mutation, crossover, and selection."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.CR = 0.9
        self.F = 0.7

        super().__init__(params)

    @property
    def CR(self) -> float:
        """Return the crossover probability."""

        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, Real):
            raise TypeError("`CR` must be a float or integer.")
        if not 0 <= CR <= 1:
            raise ValueError("`CR` must be between 0 and 1.")
        self._CR = float(CR)

    @property
    def F(self) -> float:
        """Return the differential mutation weight."""

        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, Real):
            raise TypeError("`F` must be a float or integer.")
        if not 0 <= F <= 2:
            raise ValueError("`F` must be between 0 and 2.")
        self._F = float(F)

    def compile(self, population) -> None:
        """Validate that the population can provide three distinct donors.

        Args:
            population: Population whose size is validated.

        Raises:
            ValueError: The population contains fewer than four agents.

        """

        if population.n_agents < 4:
            raise ValueError("`population.n_agents` must be at least 4.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population through mutation, crossover, and selection.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        targets = torch.arange(n, device=device)
        idx_a = _sample_excluding(targets.unsqueeze(1), n)
        idx_b = _sample_excluding(torch.stack((targets, idx_a), dim=1), n)
        idx_c = _sample_excluding(torch.stack((targets, idx_a, idx_b), dim=1), n)

        mutant = pop.positions[idx_a] + self.F * (pop.positions[idx_b] - pop.positions[idx_c])

        cr_mask = (
            torch.rand(
                n,
                pop.n_variables,
                pop.n_dimensions,
                device=device,
                dtype=pop.dtype,
            )
            < self.CR
        )
        # Force at least one mutant variable into each trial
        j_rand = torch.randint(0, pop.n_variables, (n,), device=device)
        for i in range(n):
            cr_mask[i, j_rand[i], :] = True

        trial = torch.where(cr_mask, mutant, pop.positions)
        trial = trial.clamp(min=lb, max=ub)

        trial_fitness = fn(trial)
        improved = trial_fitness < pop.fitness
        pop.positions[improved] = trial[improved]
        pop.fitness[improved] = trial_fitness[improved]
