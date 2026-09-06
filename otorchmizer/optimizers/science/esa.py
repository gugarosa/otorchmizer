# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Electro-Search Algorithm.

References:
    A. Tabari and A. Ahmad. A new optimization method: Electro-Search algorithm.
    Computers & Chemical Engineering (2017).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


def _nonzero(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


class ESA(Optimizer):
    """Electro-Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ESA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_electrons = 5
        super().__init__(params)

    @property
    def n_electrons(self) -> int:
        """Return the number of sampled electrons.

        Returns:
            int: Current electron count.

        """

        return self._n_electrons

    @n_electrons.setter
    def n_electrons(self, value: int) -> None:
        """Set the number of sampled electrons.

        Args:
            value: New electron count.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`n_electrons` must be an integer.")
        if value <= 0:
            raise ValueError("`n_electrons` must be positive.")
        self._n_electrons = value

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.D = torch.rand_like(population.positions)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ESA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        eps = torch.finfo(pop.dtype).eps

        for i in range(n):
            reference_best = pop.best_position.clone()
            levels = torch.randint(2, 6, (self.n_electrons, 1, 1), device=pop.device)
            radius = _nonzero(self.D[i]).unsqueeze(0)
            electrons = (
                pop.positions[i].unsqueeze(0)
                + (torch.rand_like(radius.expand(self.n_electrons, -1, -1)) * 2 - 1)
                * (1 - levels.to(pop.dtype).square().reciprocal())
                / radius
            )
            electrons = electrons.clamp(min=pop.lb, max=pop.ub)
            electron_fitness = fn(electrons)
            best_electron_index = electron_fitness.argmin()
            best_electron = electrons[best_electron_index]
            best_electron_fitness = electron_fitness[best_electron_index]
            if best_electron_fitness < pop.best_fitness:
                pop.best_position = best_electron.clone()
                pop.best_fitness = best_electron_fitness.clone()

            rydberg = torch.rand((), device=pop.device, dtype=pop.dtype)
            acceleration = torch.rand((), device=pop.device, dtype=pop.dtype)
            best_inverse = reference_best.square().clamp_min(eps).reciprocal()
            current_inverse = pop.positions[i].square().clamp_min(eps).reciprocal()
            self.D[i] = best_electron - reference_best + rydberg * (best_inverse - current_inverse)

            candidate = (pop.positions[i] + acceleration * self.D[i]).clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]
            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit

        pop.update_best()
