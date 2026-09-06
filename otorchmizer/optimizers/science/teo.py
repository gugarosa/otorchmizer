# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Thermal Exchange Optimization.

References:
    A. Kaveh and A. Dadras. A novel meta-heuristic optimization algorithm: Thermal exchange optimization.
    Advances in Engineering Software (2017).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class TEO(Optimizer):
    """Thermal Exchange Optimization.

    Notes:
        Fitness values must be finite and non-negative because they define the heat-transfer coefficient.
        For odd populations, the median ranked object uses itself as its environment while the remaining objects
        are paired between the better and worse halves.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the TEO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c1 = True
        self.c2 = True
        self.pro = 0.05
        self.n_TM = 4
        super().__init__(params)

    @property
    def c1(self) -> bool:
        """Return the random step-size switch.

        Returns:
            bool: Current step-size switch.

        """

        return self._c1

    @c1.setter
    def c1(self, value: bool) -> None:
        """Set the random step-size switch.

        Args:
            value: New step-size switch.

        Raises:
            TypeError: If the value is not Boolean.

        """

        if not isinstance(value, bool):
            raise TypeError("`c1` must be a bool.")
        self._c1 = value

    @property
    def c2(self) -> bool:
        """Return the randomness switch.

        Returns:
            bool: Current randomness switch.

        """

        return self._c2

    @c2.setter
    def c2(self, value: bool) -> None:
        """Set the randomness switch.

        Args:
            value: New randomness switch.

        Raises:
            TypeError: If the value is not Boolean.

        """

        if not isinstance(value, bool):
            raise TypeError("`c2` must be a bool.")
        self._c2 = value

    @property
    def pro(self) -> float:
        """Return the reset probability.

        Returns:
            float: Current reset probability.

        """

        return self._pro

    @pro.setter
    def pro(self, value: float) -> None:
        """Set the reset probability.

        Args:
            value: New reset probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`pro` must be a float or integer.")
        if not 0 <= value <= 1:
            raise ValueError("`pro` must be between 0 and 1.")
        self._pro = float(value)

    @property
    def n_TM(self) -> int:
        """Return the thermal-memory capacity.

        Returns:
            int: Current thermal-memory capacity.

        """

        return self._n_TM

    @n_TM.setter
    def n_TM(self, value: int) -> None:
        """Set the thermal-memory capacity.

        Args:
            value: New memory capacity.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`n_TM` must be an integer.")
        if value <= 0:
            raise ValueError("`n_TM` must be positive.")
        self._n_TM = value

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.environment = population.positions.clone()
        self.TM = []

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one TEO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        t = ctx.iteration / max(ctx.n_iterations, 1)
        if not torch.isfinite(pop.fitness).all() or (pop.fitness < 0).any():
            raise ValueError("`population.fitness` must contain finite non-negative values for TEO.")

        order = torch.argsort(pop.fitness)
        positions = pop.positions[order].clone()
        fitness = pop.fitness[order].clone()
        memory = self.TM or [(positions[i].clone(), fitness[i].clone()) for i in range(min(self.n_TM, n))]

        r = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        factor = float(self.c1) + float(self.c2) * (1 - t)
        pair = torch.arange(n, device=device)
        half = n // 2
        pair[:half] = torch.arange(n - half, n, device=device)
        pair[n - half :] = torch.arange(half, device=device)
        modified_environment = (1 - factor * r) * positions
        environment = modified_environment[pair]

        worst_fit = fitness[-1]
        beta = torch.zeros_like(fitness) if worst_fit == 0 else fitness / worst_fit
        candidates = environment + (positions - environment) * torch.exp(-beta.view(n, 1, 1) * t)
        for i in range(n):
            if torch.rand((), device=device) < self.pro:
                j = torch.randint(0, pop.n_variables, (), device=device).item()
                candidates[i, j] = torch.rand_like(candidates[i, j]) * (pop.ub[j] - pop.lb[j]) + pop.lb[j]

        candidates = candidates.clamp(min=pop.lb, max=pop.ub)
        candidate_fitness = fn(candidates)
        if not torch.isfinite(candidate_fitness).all() or (candidate_fitness < 0).any():
            raise ValueError("`function` must return finite non-negative values for TEO.")

        memory_positions = torch.stack([position for position, _ in memory])
        memory_fitness = torch.stack([value for _, value in memory])
        combined_positions = torch.cat((candidates, memory_positions))
        combined_fitness = torch.cat((candidate_fitness, memory_fitness))
        selected = torch.argsort(combined_fitness)[:n]
        memory_selected = torch.argsort(combined_fitness)[: min(self.n_TM, combined_fitness.numel())]

        self.environment = environment
        self.TM = [(combined_positions[index].clone(), combined_fitness[index].clone()) for index in memory_selected]
        pop.positions = combined_positions[selected]
        pop.fitness = combined_fitness[selected]
        pop.update_best()
