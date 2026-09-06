# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Henry Gas Solubility Optimization.

References:
    F. Hashim et al. Henry gas solubility optimization: A novel physics-based algorithm.
    Future Generation Computer Systems (2019).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class HGSO(Optimizer):
    """Henry Gas Solubility Optimization.

    Notes:
        Fitness values must be finite and non-negative because the gas-pressure coefficient uses fitness ratios.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the HGSO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_clusters = 2
        self.alpha = 1.0
        self.beta = 1.0
        self.K = 1.0
        self.l1 = 0.0005
        self.l2 = 100.0
        self.l3 = 0.001
        super().__init__(params)

    @property
    def n_clusters(self) -> int:
        """Return the number of gas clusters.

        Returns:
            int: Current cluster count.

        """

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int) -> None:
        """Set the number of gas clusters.

        Args:
            value: New cluster count.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`n_clusters` must be an integer.")
        if value <= 0:
            raise ValueError("`n_clusters` must be positive.")
        self._n_clusters = value

    @property
    def l1(self) -> float:
        """Return the Henry-coefficient scale.

        Returns:
            float: Current coefficient scale.

        """

        return self._l1

    @l1.setter
    def l1(self, value: float) -> None:
        """Set the Henry-coefficient scale.

        Args:
            value: New coefficient scale.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`l1` must be a float or integer.")
        if value < 0:
            raise ValueError("`l1` must be non-negative.")
        self._l1 = float(value)

    @property
    def l2(self) -> float:
        """Return the pressure scale.

        Returns:
            float: Current pressure scale.

        """

        return self._l2

    @l2.setter
    def l2(self, value: float) -> None:
        """Set the pressure scale.

        Args:
            value: New pressure scale.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`l2` must be a float or integer.")
        if value <= 0:
            raise ValueError("`l2` must be positive.")
        self._l2 = float(value)

    @property
    def l3(self) -> float:
        """Return the Henry-schedule scale.

        Returns:
            float: Current schedule scale.

        """

        return self._l3

    @l3.setter
    def l3(self, value: float) -> None:
        """Set the Henry-schedule scale.

        Args:
            value: New schedule scale.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`l3` must be a float or integer.")
        if value < 0:
            raise ValueError("`l3` must be non-negative.")
        self._l3 = float(value)

    @property
    def alpha(self) -> float:
        """Return the gas-influence coefficient.

        Returns:
            float: Current gas-influence coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the gas-influence coefficient.

        Args:
            value: New gas-influence coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        if value < 0:
            raise ValueError("`alpha` must be non-negative.")
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        """Return the fitness-pressure coefficient.

        Returns:
            float: Current fitness-pressure coefficient.

        """

        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the fitness-pressure coefficient.

        Args:
            value: New fitness-pressure coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        if value < 0:
            raise ValueError("`beta` must be non-negative.")
        self._beta = float(value)

    @property
    def K(self) -> float:
        """Return the solubility coefficient.

        Returns:
            float: Current solubility coefficient.

        """

        return self._K

    @K.setter
    def K(self, value: float) -> None:
        """Set the solubility coefficient.

        Args:
            value: New solubility coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`K` must be a float or integer.")
        if value < 0:
            raise ValueError("`K` must be non-negative.")
        self._K = float(value)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if self.n_clusters > population.n_agents:
            raise ValueError("`n_clusters` must not exceed `population.n_agents`.")

        self.coeff = torch.rand(self.n_clusters, device=population.device, dtype=population.dtype) * self.l1
        self.pressure = torch.rand(population.n_agents, device=population.device, dtype=population.dtype) * self.l2
        self.constant = torch.rand(self.n_clusters, device=population.device, dtype=population.dtype) * self.l3

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one HGSO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        if not torch.isfinite(pop.fitness).all() or (pop.fitness < 0).any():
            raise ValueError("`population.fitness` must contain finite non-negative values for HGSO.")

        temperature = torch.exp(pop.positions.new_tensor(-ctx.iteration / max(ctx.n_iterations, 1)))
        schedule = -self.constant * (temperature.reciprocal() - 1 / 298.15)
        coefficient = self.coeff * torch.exp(schedule)

        source_positions = pop.positions.clone()
        candidates = source_positions.clone()
        best_position = pop.best_position.clone()
        clusters = torch.tensor_split(torch.arange(n, device=device), self.n_clusters)
        for cluster_index, indices in enumerate(clusters):
            cluster_best = indices[pop.fitness[indices].argmin()]
            cluster_best_position = source_positions[cluster_best].clone()

            for index in indices:
                solubility = self.K * coefficient[cluster_index] * self.pressure[index]
                gamma = self.beta * torch.exp(-(pop.best_fitness + 0.05) / (pop.fitness[index] + 0.05))
                direction = -1.0 if torch.rand((), device=device) < 0.5 else 1.0
                r = torch.rand((), device=device, dtype=pop.dtype)
                candidates[index] = (
                    source_positions[index]
                    + direction * r * gamma * (cluster_best_position - source_positions[index])
                    + direction * r * self.alpha * (solubility * best_position - source_positions[index])
                )

        candidates = candidates.clamp(min=pop.lb, max=pop.ub)
        candidate_fitness = fn(candidates)
        if not torch.isfinite(candidate_fitness).all() or (candidate_fitness < 0).any():
            raise ValueError("`function` must return finite non-negative values for HGSO.")

        self.coeff = coefficient
        pop.positions = candidates
        pop.fitness = candidate_fitness
        pop.update_best()

        fraction = 0.1 + 0.1 * torch.rand((), device=device, dtype=pop.dtype)
        n_replace = int(n * fraction.item())
        if n_replace:
            worst = torch.argsort(pop.fitness, descending=True)[:n_replace]
            replacement_positions = torch.rand_like(pop.positions[worst]) * (pop.ub - pop.lb) + pop.lb
            replacement_fitness = fn(replacement_positions)
            if not torch.isfinite(replacement_fitness).all() or (replacement_fitness < 0).any():
                raise ValueError("`function` must return finite non-negative values for HGSO.")

            pop.positions[worst] = replacement_positions
            pop.fitness[worst] = replacement_fitness
            pop.update_best()
