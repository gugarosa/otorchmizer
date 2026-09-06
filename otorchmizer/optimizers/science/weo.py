# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Water Evaporation Optimization.

References:
    A. Kaveh and T. Bakhshpoori.
    Water Evaporation Optimization: A novel physically inspired optimization algorithm.
    Computers & Structures (2016).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class WEO(Optimizer):
    """Water Evaporation Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the WEO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.E_min = -3.5
        self.E_max = -0.5
        self.theta_min = -torch.pi / 3.6
        self.theta_max = -torch.pi / 9
        super().__init__(params)

    @property
    def E_min(self) -> float:
        """Return the minimum substrate energy.

        Returns:
            float: Current minimum substrate energy.

        """

        return self._E_min

    @E_min.setter
    def E_min(self, value: float) -> None:
        """Set the minimum substrate energy.

        Args:
            value: New minimum substrate energy.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`E_min` must be a float or integer.")
        self._E_min = float(value)

    @property
    def E_max(self) -> float:
        """Return the maximum substrate energy.

        Returns:
            float: Current maximum substrate energy.

        """

        return self._E_max

    @E_max.setter
    def E_max(self, value: float) -> None:
        """Set the maximum substrate energy.

        Args:
            value: New maximum substrate energy.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is below the minimum.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`E_max` must be a float or integer.")
        if value < self.E_min:
            raise ValueError("`E_max` must be greater than or equal to `E_min`.")
        self._E_max = float(value)

    @property
    def theta_min(self) -> float:
        """Return the minimum contact angle.

        Returns:
            float: Current minimum contact angle.

        """

        return self._theta_min

    @theta_min.setter
    def theta_min(self, value: float) -> None:
        """Set the minimum contact angle.

        Args:
            value: New minimum contact angle.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`theta_min` must be a float or integer.")
        self._theta_min = float(value)

    @property
    def theta_max(self) -> float:
        """Return the maximum contact angle.

        Returns:
            float: Current maximum contact angle.

        """

        return self._theta_max

    @theta_max.setter
    def theta_max(self, value: float) -> None:
        """Set the maximum contact angle.

        Args:
            value: New maximum contact angle.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is below the minimum.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`theta_max` must be a float or integer.")
        if value < self.theta_min:
            raise ValueError("`theta_max` must be greater than or equal to `theta_min`.")
        self._theta_max = float(value)

    def compile(self, population) -> None:
        """Validate the population required by pairwise evaporation steps.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            ValueError: If fewer than two agents are available.

        """

        if population.n_agents < 2:
            raise ValueError("`population.n_agents` must be at least 2 for WEO.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one WEO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best_fit = pop.fitness.min()
        worst_fit = pop.fitness.max()
        denominator = (worst_fit - best_fit).clamp_min(torch.finfo(pop.dtype).eps)

        for i in range(n):
            normalized = (pop.fitness[i] - best_fit) / denominator
            if ctx.iteration <= max(ctx.n_iterations, 1) / 2:
                substrate_energy = (self.E_max - self.E_min) * normalized + self.E_min
                probability = torch.exp(substrate_energy)
            else:
                theta = (self.theta_max - self.theta_min) * normalized + self.theta_min
                cosine = torch.cos(theta)
                base = (2 / 3 + cosine**3 / 3 - cosine).clamp_min(torch.finfo(pop.dtype).eps)
                probability = (1 / 2.6) * base.pow(-2 / 3) * (1 - cosine)

            mask = (torch.rand_like(pop.positions[i]) < probability).to(pop.dtype)
            pair = torch.randperm(n, device=device)[:2]
            step = torch.rand((), device=device, dtype=pop.dtype) * (pop.positions[pair[0]] - pop.positions[pair[1]])
            candidate = (pop.positions[i] + step * mask).clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]
            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit

        pop.update_best()
