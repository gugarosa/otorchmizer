# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Lightning Search Algorithm.

References:
    H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm.
    Applied Soft Computing (2015).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class LSA(Optimizer):
    """Lightning Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the LSA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.max_time = 10
        self.E = 2.05
        self.p_fork = 0.01
        super().__init__(params)

    @property
    def max_time(self) -> int:
        """Return the maximum channel time.

        Returns:
            int: Current maximum channel time.

        """

        return self._max_time

    @max_time.setter
    def max_time(self, value: int) -> None:
        """Set the maximum channel time.

        Args:
            value: New maximum channel time.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`max_time` must be an integer.")
        if value <= 0:
            raise ValueError("`max_time` must be positive.")
        self._max_time = value

    @property
    def E(self) -> float:
        """Return the initial energy.

        Returns:
            float: Current initial energy.

        """

        return self._E

    @E.setter
    def E(self, value: float) -> None:
        """Set the initial energy.

        Args:
            value: New initial energy.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`E` must be a float or integer.")
        if value < 0:
            raise ValueError("`E` must be non-negative.")
        self._E = float(value)

    @property
    def p_fork(self) -> float:
        """Return the forking probability.

        Returns:
            float: Current forking probability.

        """

        return self._p_fork

    @p_fork.setter
    def p_fork(self, value: float) -> None:
        """Set the forking probability.

        Args:
            value: New forking probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`p_fork` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`p_fork` must be between 0 and 1.")
        self._p_fork = float(value)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.time = 0
        random_direction = torch.rand_like(population.positions[0]) * 2 - 1
        self.direction = torch.where(
            random_direction < 0, -torch.ones_like(random_direction), torch.ones_like(random_direction)
        )

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one LSA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        t = ctx.iteration / max(ctx.n_iterations, 1)

        self.time += 1
        if self.time >= self.max_time:
            worst_idx = pop.fitness.argmax()
            pop.positions[worst_idx] = pop.best_position.clone()
            pop.fitness[worst_idx] = pop.best_fitness.clone()
            self.time = 0

        order = torch.argsort(pop.fitness)
        pop.positions = pop.positions[order]
        pop.fitness = pop.fitness[order]
        best_position = pop.positions[0].clone()

        for j in range(pop.n_variables):
            shake = best_position.clone()
            shake[j] += self.direction[j] * 0.005 * (pop.ub[j] - pop.lb[j])
            shake = shake.clamp(min=pop.lb, max=pop.ub)
            shake_fitness = fn(shake.unsqueeze(0))[0]
            if shake_fitness < pop.best_fitness:
                pop.best_position = shake.clone()
                pop.best_fitness = shake_fitness.clone()
            if shake_fitness > pop.fitness[0]:
                self.direction[j] *= -1

        energy = self.E - 2 * torch.exp(pop.positions.new_tensor(-5 * (1 - t)))
        for i in range(n):
            candidate = pop.positions[i].clone()
            distance = pop.positions[i] - best_position
            zero = distance == 0
            below = distance < 0
            exponential = -torch.log(torch.rand_like(distance).clamp_min(torch.finfo(pop.dtype).tiny)) * distance.abs()
            candidate = torch.where(zero, candidate + self.direction * torch.randn_like(candidate) * energy, candidate)
            candidate = torch.where(below & ~zero, candidate + exponential, candidate)
            candidate = torch.where(~below & ~zero, candidate - exponential, candidate)
            candidate = candidate.clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]

            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit
                if torch.rand((), device=pop.device) < self.p_fork:
                    fork = torch.rand_like(candidate) * (pop.ub - pop.lb) + pop.lb
                    fork_fit = fn(fork.unsqueeze(0))[0]
                    if fork_fit < pop.fitness[i]:
                        pop.positions[i] = fork
                        pop.fitness[i] = fork_fit

        pop.update_best()
