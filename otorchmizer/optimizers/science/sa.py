# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Simulated Annealing.

References:
    S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi.
    Optimization by simulated annealing.
    Science (1983).
"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class SA(Optimizer):
    """Simulated Annealing.

    Notes:
        Uses temperature-controlled Metropolis acceptance for candidate moves.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.T = 100.0
        self.beta = 0.999

        super().__init__(params)

    @property
    def T(self) -> float:
        """Return the temperature.

        Returns:
            float: Current temperature.

        """

        return self._T

    @T.setter
    def T(self, T: float) -> None:
        """Set the temperature.

        Args:
            T: New value for the temperature.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(T, (float, int)):
            raise TypeError("`T` must be a float or integer.")
        self._T = T

    @property
    def beta(self) -> float:
        """Return the beta coefficient.

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

        """

        if not isinstance(beta, (float, int)):
            raise TypeError("`beta` must be a float or integer.")
        self._beta = beta

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one SA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents

        # Add Gaussian noise
        noise = torch.randn_like(pop.positions) * 0.1
        new_positions = pop.positions + noise
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        new_positions = new_positions.clamp(min=lb, max=ub)

        new_fitness = fn(new_positions)

        # Metropolis acceptance criterion
        delta = new_fitness - pop.fitness
        accept_prob = torch.exp(-delta / max(self.T, 1e-10))
        r = torch.rand(n, device=device)
        accept = (new_fitness < pop.fitness) | (r < accept_prob)

        pop.positions[accept] = new_positions[accept]
        pop.fitness[accept] = new_fitness[accept]

        # Cool temperature
        self.T *= self.beta
