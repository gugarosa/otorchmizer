# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Harmony Search and variants.

References:
    Z. W. Geem, J. H. Kim, and G. V. Loganathan.
    A new heuristic optimization algorithm: harmony search.
    Simulation (2001).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

import otorchmizer.utils.constant as c
from otorchmizer.core.optimizer import Optimizer, UpdateContext


class HS(Optimizer):
    """Apply memory consideration, pitch adjustment, and random search."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.HMCR = 0.7
        self.PAR = 0.7
        self.bw = 1.0

        super().__init__(params)

    @property
    def HMCR(self) -> float:
        """Return the harmony-memory consideration rate."""

        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR: float) -> None:
        if not isinstance(HMCR, Real):
            raise TypeError("`HMCR` must be a float or integer.")
        if not 0 <= HMCR <= 1:
            raise ValueError("`HMCR` must be between 0 and 1.")
        self._HMCR = float(HMCR)

    @property
    def PAR(self) -> float:
        """Return the pitch-adjustment rate."""

        return self._PAR

    @PAR.setter
    def PAR(self, PAR: float) -> None:
        if not isinstance(PAR, Real):
            raise TypeError("`PAR` must be a float or integer.")
        if not 0 <= PAR <= 1:
            raise ValueError("`PAR` must be between 0 and 1.")
        self._PAR = float(PAR)

    @property
    def bw(self) -> float:
        """Return the pitch-adjustment bandwidth."""

        return self._bw

    @bw.setter
    def bw(self, bw: float) -> None:
        if not isinstance(bw, Real):
            raise TypeError("`bw` must be a float or integer.")
        if bw < 0:
            raise ValueError("`bw` must be non-negative.")
        self._bw = float(bw)

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        n = pop.n_agents
        new_pos = torch.zeros(
            pop.n_variables,
            pop.n_dimensions,
            device=device,
            dtype=pop.dtype,
        )
        lb = pop.lb
        ub = pop.ub

        for j in range(pop.n_variables):
            r1 = torch.rand(1, device=device, dtype=pop.dtype).item()
            if r1 < self.HMCR:
                idx = torch.randint(0, n, (1,), device=device).item()
                new_pos[j] = pop.positions[idx, j]
                r2 = torch.rand(1, device=device, dtype=pop.dtype).item()
                if r2 < self.PAR:
                    r3 = 2 * torch.rand(1, device=device, dtype=pop.dtype).item() - 1
                    new_pos[j] = new_pos[j] + r3 * self.bw
            else:
                new_pos[j] = torch.rand(pop.n_dimensions, device=device, dtype=pop.dtype) * (ub[j] - lb[j]) + lb[j]

        return new_pos

    def update(self, ctx: UpdateContext) -> None:
        """Generate one harmony and replace the worst agent when improved.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device

        new_pos = self._generate_new_harmony(pop, device)
        lb = pop.lb
        ub = pop.ub
        new_pos = new_pos.clamp(min=lb, max=ub)

        new_fit = fn(new_pos.unsqueeze(0))[0]

        worst_idx = pop.fitness.argmax()
        if new_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = new_pos
            pop.fitness[worst_idx] = new_fit


class IHS(HS):
    """Apply Harmony Search with adaptive pitch rate and bandwidth."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.PAR_min = 0.0
        self.PAR_max = 1.0
        self.bw_min = 1.0
        self.bw_max = 10.0

        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Adapt pitch parameters and generate one harmony.

        Args:
            ctx: Current optimization state and objective.

        """

        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        self.PAR = self.PAR_min + (self.PAR_max - self.PAR_min) / T * t
        self.bw = self.bw_max * math.exp((math.log(self.bw_min / (self.bw_max + c.EPSILON)) / T) * t)

        super().update(ctx)


class GHS(IHS):
    """Apply Improved Harmony Search with global-best pitch adjustment."""

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        n = pop.n_agents
        new_pos = torch.zeros(
            pop.n_variables,
            pop.n_dimensions,
            device=device,
            dtype=pop.dtype,
        )
        lb = pop.lb
        ub = pop.ub

        for j in range(pop.n_variables):
            r1 = torch.rand(1, device=device, dtype=pop.dtype).item()
            if r1 < self.HMCR:
                idx = torch.randint(0, n, (1,), device=device).item()
                new_pos[j] = pop.positions[idx, j]
                r2 = torch.rand(1, device=device, dtype=pop.dtype).item()
                if r2 < self.PAR:
                    z = torch.randint(0, pop.n_variables, (1,), device=device).item()
                    new_pos[j] = pop.best_position[z]
            else:
                new_pos[j] = torch.rand(pop.n_dimensions, device=device, dtype=pop.dtype) * (ub[j] - lb[j]) + lb[j]

        return new_pos


class SGHS(HS):
    """Apply Global-Best Harmony Search with learned sampling parameters."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.LP = 100
        self.HMCRm = 0.98
        self.PARm = 0.9
        self.bw_min = 1.0
        self.bw_max = 10.0

        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize learning-period histories.

        Args:
            population: Population accepted for the shared compile contract.

        """

        self.lp = 1
        self.HMCR_history = []
        self.PAR_history = []

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        new_pos = pop.best_position.clone()
        lb = pop.lb
        ub = pop.ub

        for j in range(pop.n_variables):
            if torch.rand(1, device=device, dtype=pop.dtype).item() < self.HMCR:
                adjustment = (2 * torch.rand(1, device=device, dtype=pop.dtype).item() - 1) * self.bw
                new_pos[j] += adjustment
                if torch.rand(1, device=device, dtype=pop.dtype).item() < self.PAR:
                    new_pos[j] = pop.best_position[j]
            else:
                new_pos[j] = torch.rand(pop.n_dimensions, device=device, dtype=pop.dtype) * (ub[j] - lb[j]) + lb[j]

        return new_pos

    def update(self, ctx: UpdateContext) -> None:
        """Sample adaptive parameters and generate one harmony.

        Args:
            ctx: Current optimization state and objective.

        """

        device = ctx.space.population.device
        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        dtype = ctx.space.population.dtype
        hmcr = min(max(torch.randn(1, device=device, dtype=dtype).item() * 0.01 + self.HMCRm, 0), 1)
        par = min(max(torch.randn(1, device=device, dtype=dtype).item() * 0.05 + self.PARm, 0), 1)
        self.HMCR = hmcr
        self.PAR = par
        self.HMCR_history.append(hmcr)
        self.PAR_history.append(par)

        if t < T / 2:
            self.bw = self.bw_max - (self.bw_max - self.bw_min) * (2 * t / T)
        else:
            self.bw = self.bw_min

        HS.update(self, ctx)

        if self.lp >= self.LP:
            if self.HMCR_history:
                self.HMCRm = sum(self.HMCR_history) / len(self.HMCR_history)
            if self.PAR_history:
                self.PARm = sum(self.PAR_history) / len(self.PAR_history)
            self.lp = 1
            self.HMCR_history = []
            self.PAR_history = []
        else:
            self.lp += 1


class NGHS(HS):
    """Apply Harmony Search with best-worst generation and mutation."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        """

        self.pm = 0.1

        super().__init__(params)

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        lb = pop.lb
        ub = pop.ub

        worst_idx = pop.fitness.argmax()
        worst = pop.positions[worst_idx]

        # Correct the inherited implementation to reflect the worst harmony through the best
        new_pos_range = 2 * pop.best_position - worst
        new_pos_range = new_pos_range.clamp(min=lb, max=ub)

        r1 = torch.rand(
            pop.n_variables,
            pop.n_dimensions,
            device=device,
            dtype=pop.dtype,
        )
        new_pos = worst + r1 * (new_pos_range - worst)

        if torch.rand(1, device=device, dtype=pop.dtype).item() < self.pm:
            j = torch.randint(0, pop.n_variables, (1,), device=device).item()
            new_pos[j] = torch.rand(pop.n_dimensions, device=device, dtype=pop.dtype) * (ub[j] - lb[j]) + lb[j]

        return new_pos

    def update(self, ctx: UpdateContext) -> None:
        """Generate a harmony and unconditionally replace the worst harmony.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        new_pos = self._generate_new_harmony(pop, pop.device)
        new_pos = new_pos.clamp(min=pop.lb, max=pop.ub)
        new_fit = ctx.function(new_pos.unsqueeze(0))[0]

        if new_fit < pop.best_fitness:
            pop.best_position = new_pos.clone()
            pop.best_fitness = new_fit.clone()

        worst_idx = pop.fitness.argmax()
        pop.positions[worst_idx] = new_pos
        pop.fitness[worst_idx] = new_fit


class GOGHS(NGHS):
    """Apply Novel Global-Best Harmony Search with opposition learning."""

    def update(self, ctx: UpdateContext) -> None:
        """Generate direct and opposite harmonies and retain the better one.

        Args:
            ctx: Current optimization state and objective.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device

        new_pos = self._generate_new_harmony(pop, device)
        lb = pop.lb
        ub = pop.ub
        new_pos = new_pos.clamp(min=lb, max=ub)
        new_fit = fn(new_pos.unsqueeze(0))[0]

        A = pop.positions.min(dim=0).values
        B = pop.positions.max(dim=0).values
        k = torch.rand(1, device=device, dtype=pop.dtype)
        opp_pos = k * (A + B) - new_pos
        opp_pos = opp_pos.clamp(min=lb, max=ub)
        opp_fit = fn(opp_pos.unsqueeze(0))[0]

        if opp_fit < new_fit:
            new_pos = opp_pos
            new_fit = opp_fit

        worst_idx = pop.fitness.argmax()
        if new_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = new_pos
            pop.fitness[worst_idx] = new_fit
