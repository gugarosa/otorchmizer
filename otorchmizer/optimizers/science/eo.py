# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Equilibrium Optimizer.

References:
    A. Faramarzi et al.
    Equilibrium optimizer: A novel optimization algorithm.
    Knowledge-Based Systems (2020).
"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class EO(Optimizer):
    """Equilibrium Optimizer.

    Notes:
        Applies concentration-based movement with generation-rate control.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the EO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.a1 = 2.0
        self.a2 = 1.0
        self.GP = 0.5
        self.V = 1.0

        super().__init__(params)

    @property
    def a1(self) -> float:
        """Return the exploration coefficient.

        Returns:
            float: Current exploration coefficient.

        """

        return self._a1

    @a1.setter
    def a1(self, a1: float) -> None:
        """Set the exploration coefficient.

        Args:
            a1: New value for the exploration coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(a1, (float, int)):
            raise TypeError("`a1` must be a float or integer.")
        self._a1 = a1

    @property
    def a2(self) -> float:
        """Return the exploitation coefficient.

        Returns:
            float: Current exploitation coefficient.

        """

        return self._a2

    @a2.setter
    def a2(self, a2: float) -> None:
        """Set the exploitation coefficient.

        Args:
            a2: New value for the exploitation coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(a2, (float, int)):
            raise TypeError("`a2` must be a float or integer.")
        self._a2 = a2

    @property
    def GP(self) -> float:
        """Return the generation probability.

        Returns:
            float: Current generation probability.

        """

        return self._GP

    @GP.setter
    def GP(self, GP: float) -> None:
        """Set the generation probability.

        Args:
            GP: New value for the generation probability.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(GP, (float, int)):
            raise TypeError("`GP` must be a float or integer.")
        if not 0 <= GP <= 1:
            raise ValueError("`GP` must be between 0 and 1.")
        self._GP = GP

    @property
    def V(self) -> float:
        """Return the volume coefficient.

        Returns:
            float: Current volume coefficient.

        """

        return self._V

    @V.setter
    def V(self, V: float) -> None:
        """Set the volume coefficient.

        Args:
            V: New value for the volume coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the supplied value is outside its valid range.

        """

        if not isinstance(V, (float, int)):
            raise TypeError("`V` must be a float or integer.")
        if V <= 0:
            raise ValueError("`V` must be positive.")
        self._V = V

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """
        shape = (population.n_variables, population.n_dimensions)
        self.C = [population.positions.new_zeros(shape) for _ in range(4)]
        self.C_fit = [population.fitness.new_full((), torch.inf) for _ in range(4)]

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one EO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        # Update equilibrium pool (top-4)
        for i in range(n):
            for k in range(4):
                if pop.fitness[i] < self.C_fit[k]:
                    # Shift down
                    for j in range(3, k, -1):
                        self.C[j] = self.C[j - 1].clone()
                        self.C_fit[j] = self.C_fit[j - 1].clone()
                    self.C[k] = pop.positions[i].clone()
                    self.C_fit[k] = pop.fitness[i].clone()
                    break

        # Average concentration
        C_avg = sum(self.C) / 4
        C_pool = self.C + [C_avg]

        # Time factor
        time = (1 - t / T) ** (self.a2 * t / T)

        for i in range(n):
            # Random equilibrium from pool
            idx = torch.randint(0, 5, (1,), device=device).item()
            C_eq = C_pool[idx]

            r = torch.rand_like(pop.positions[i])
            lam = torch.rand_like(pop.positions[i])

            # Exponential term
            F = self.a1 * torch.sign(r - 0.5) * (torch.exp(-lam * time) - 1)

            # Generation probability
            r_GP = torch.rand(1, device=device)
            GCP = 0.5 * r_GP if r_GP >= self.GP else pop.positions.new_zeros(())

            G = GCP * (C_eq - lam * pop.positions[i]) * F

            pop.positions[i] = C_eq + (pop.positions[i] - C_eq) * F + (G / (lam * self.V + 1e-10)) * (1 - F)

        pop.positions = pop.positions.clamp(min=lb, max=ub)
