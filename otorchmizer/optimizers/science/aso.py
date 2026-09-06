# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Atom Search Optimization.

References:
    W. Zhao, L. Wang, and Z. Zhang.
    Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem.
    Knowledge-Based Systems (2019).
"""

from __future__ import annotations

from math import exp, pi, sin, sqrt
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class ASO(Optimizer):
    """Atom Search Optimization.

    Notes:
        Interatomic Lennard-Jones forces and the best-atom constraint share the
        gravitational decay and inverse-mass scaling in the acceleration equation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ASO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 50.0
        self.beta = 0.2
        super().__init__(params)

    @property
    def alpha(self) -> float:
        """Return the interatomic potential depth weight.

        Returns:
            float: Current alpha coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set the alpha coefficient.

        Args:
            alpha: New value for the alpha coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(alpha, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Return the best-atom constraint weight.

        Returns:
            float: Current beta coefficient.

        """

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        """Set the beta coefficient.

        Args:
            beta: New value for the beta coefficient.

        Raises:
            TypeError: If the supplied value has an invalid type.
            ValueError: If the constraint weight is outside [0, 1].

        """

        if not isinstance(beta, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        if not 0 <= beta <= 1:
            raise ValueError("`beta` must be between 0 and 1.")
        self._beta = beta

    def compile(self, population: Population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.velocity = torch.zeros_like(population.positions)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ASO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        Raises:
            ValueError: The population contains non-finite fitness values.

        """

        pop = ctx.space.population
        n = pop.n_agents
        if not torch.isfinite(pop.fitness).all():
            raise ValueError("`population.fitness` must be finite for ASO mass calculation.")

        tiny = torch.finfo(pop.dtype).tiny
        fitness = pop.fitness / pop.fitness.abs().max().clamp_min(tiny)
        spread = (fitness.max() - fitness.min()).clamp_min(tiny)
        mass = torch.exp(-(fitness - fitness.min()) / spread)
        mass /= mass.sum()

        progress = ctx.iteration / max(ctx.n_iterations, 1)
        n_best = min(n, max(1, int(n - (n - 2) * sqrt(progress))))
        neighbors = pop.positions[torch.argsort(pop.fitness)[:n_best]]
        centroid = neighbors.mean(dim=0)
        mean_distance = torch.linalg.vector_norm(pop.positions - centroid, dim=(1, 2)).clamp_min(tiny)
        minimum_ratio = 1.1 + 0.1 * sin((ctx.iteration + 1) / max(ctx.n_iterations, 1) * pi / 2)

        force = torch.zeros_like(pop.positions)
        for neighbor in neighbors:
            displacement = neighbor - pop.positions
            radius = torch.linalg.vector_norm(displacement, dim=(1, 2)).clamp_min(tiny)
            ratio = (radius / mean_distance).clamp(min=minimum_ratio, max=1.24)
            potential = (1 - progress) ** 3 * (6 * ratio.pow(-7) - 12 * ratio.pow(-13))
            weight = torch.rand(n, 1, 1, device=pop.device, dtype=pop.dtype)
            force += weight * (potential / radius).view(n, 1, 1) * displacement

        attraction = pop.best_position - pop.positions
        acceleration = exp(-20 * progress) * (self.alpha * force + self.beta * attraction) / mass.view(n, 1, 1)
        inertia = torch.rand(n, 1, 1, device=pop.device, dtype=pop.dtype)
        self.velocity = inertia * self.velocity + acceleration
        pop.positions = pop.positions + self.velocity
