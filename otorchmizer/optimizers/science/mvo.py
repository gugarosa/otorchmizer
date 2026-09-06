# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Multi-Verse Optimizer.

References:
    S. Mirjalili, S. M. Mirjalili and A. Hatamlou.
    Multi-verse optimizer: a nature-inspired algorithm for global optimization.
    Neural Computing and Applications (2016).

"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class MVO(Optimizer):
    """Multi-Verse Optimizer.

    Notes:
        Combines white-hole, wormhole, and black-hole search mechanisms.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the MVO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.WEP_min = 0.2
        self.WEP_max = 1.0
        self.p = 6.0

        super().__init__(params)

    @property
    def WEP_min(self) -> float:
        """Return the minimum wormhole existence probability.

        Returns:
            float: Current minimum wormhole existence probability.

        """

        return self._WEP_min

    @WEP_min.setter
    def WEP_min(self, WEP_min: float) -> None:
        """Set the minimum wormhole existence probability.

        Args:
            WEP_min: New value for the minimum wormhole existence probability.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(WEP_min, (float, int)):
            raise TypeError("`WEP_min` must be a float or integer.")
        self._WEP_min = WEP_min

    @property
    def WEP_max(self) -> float:
        """Return the maximum wormhole existence probability.

        Returns:
            float: Current maximum wormhole existence probability.

        """

        return self._WEP_max

    @WEP_max.setter
    def WEP_max(self, WEP_max: float) -> None:
        """Set the maximum wormhole existence probability.

        Args:
            WEP_max: New value for the maximum wormhole existence probability.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(WEP_max, (float, int)):
            raise TypeError("`WEP_max` must be a float or integer.")
        self._WEP_max = WEP_max

    @property
    def p(self) -> float:
        """Return the exploitation accuracy.

        Returns:
            float: Current exploitation accuracy.

        """

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        """Set the exploitation accuracy.

        Args:
            p: New value for the exploitation accuracy.

        Raises:
            TypeError: If the supplied value has an invalid type.

        """

        if not isinstance(p, (float, int)):
            raise TypeError("`p` must be a float or integer.")
        self._p = p

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one MVO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration + 1
        T = max(ctx.n_iterations, 1)

        WEP = self.WEP_min + t * (self.WEP_max - self.WEP_min) / T
        TDR = 1 - (t ** (1 / self.p)) / (T ** (1 / self.p))

        # Normalize fitness for roulette
        norm_fit = pop.fitness - pop.fitness.min()
        norm_fit = norm_fit / (norm_fit.sum() + 1e-10)

        new_positions = pop.positions.clone()

        for i in range(n):
            for j in range(pop.n_variables):
                r1 = torch.rand(1, device=device).item()

                if r1 < norm_fit[i]:
                    # White hole
                    k = torch.multinomial(1 - norm_fit + 1e-10, 1).item()
                    new_positions[i, j] = pop.positions[k, j]

                r2 = torch.rand(1, device=device).item()
                if r2 < WEP:
                    r3 = torch.rand(1, device=device).item()
                    displacement = lb.squeeze(0)[j] + torch.rand((), device=device, dtype=pop.dtype) * (
                        ub.squeeze(0)[j] - lb.squeeze(0)[j]
                    )
                    if r3 < 0.5:
                        new_positions[i, j] = best.squeeze(0)[j] + TDR * displacement
                    else:
                        new_positions[i, j] = best.squeeze(0)[j] - TDR * displacement

        pop.positions = new_positions.clamp(min=lb, max=ub)
