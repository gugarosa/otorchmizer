# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Darcy Optimization Algorithm.

References:
    F. A. Hashim et al.
    Darcy Optimization Algorithm: A New Darcy's Law-based Optimizer.
    IEEE Access (2023).
"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class DOA(Optimizer):
    """Darcy Optimization Algorithm.

    Notes:
        Uses chaotic-map values to model Darcy-inspired flow dynamics.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the DOA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        logger.info("Overriding class: Optimizer -> DOA.")

        self.r = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def r(self) -> float:
        """Return the chaos multiplier.

        Returns:
            float: Current chaos multiplier.

        """

        return self._r

    @r.setter
    def r(self, r: float) -> None:
        """Set the chaos multiplier.

        Args:
            r: New value for the chaos multiplier.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(r, (float, int)):
            raise e.TypeError("`r` must be a float or integer.")
        self._r = r

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.chaotic_map = population.positions.new_zeros(population.n_agents, population.n_variables)

    def _calculate_chaotic_map(self, lb_val: float, ub_val: float, device: torch.device) -> torch.Tensor:
        r1 = torch.rand(1, device=device) * (ub_val - lb_val) + lb_val
        c_map = self.r * r1 * (1 - r1) + ((4 - self.r) * torch.sin(torch.pi * r1)) / 4
        return c_map

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one DOA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb
        ub = pop.ub

        for i in range(n):
            for j in range(pop.n_variables):
                lb_val = lb[j, 0].item()
                ub_val = ub[j, 0].item()

                c_map = self._calculate_chaotic_map(lb_val, ub_val, device)
                old_cmap = self.chaotic_map[i, j]

                diff = c_map - old_cmap
                if diff.abs() < 1e-10:
                    diff = torch.tensor(1e-10, device=device)

                update = (2 * (best[0, j, :] - pop.positions[i, j, :]) / diff) * (ub_val - lb_val) / n
                pop.positions[i, j, :] = pop.positions[i, j, :] + update

                self.chaotic_map[i, j] = c_map.item()

                # Boundary check
                out_of_bounds = (pop.positions[i, j, :] < lb_val) | (pop.positions[i, j, :] > ub_val)
                if out_of_bounds.any():
                    pop.positions[i, j, :] = best[0, j, :] * c_map

        lb_exp = lb.unsqueeze(0)
        ub_exp = ub.unsqueeze(0)
        pop.positions = pop.positions.clamp(min=lb_exp, max=ub_exp)
