"""Harmony Search and variants.

References:
    Z. W. Geem, J. H. Kim, and G. V. Loganathan.
    A new heuristic optimization algorithm: harmony search.
    Simulation (2001).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class HS(Optimizer):
    """Harmony Search.

    Memory consideration, pitch adjustment, and random search.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> HS.")

        self.HMCR = 0.7
        self.PAR = 0.7
        self.bw = 1.0

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def HMCR(self) -> float:
        return self._HMCR

    @HMCR.setter
    def HMCR(self, HMCR: float) -> None:
        if not isinstance(HMCR, (float, int)):
            raise e.TypeError("`HMCR` should be a float or integer")
        if not 0 <= HMCR <= 1:
            raise e.ValueError("`HMCR` should be between 0 and 1")
        self._HMCR = HMCR

    @property
    def PAR(self) -> float:
        return self._PAR

    @PAR.setter
    def PAR(self, PAR: float) -> None:
        if not isinstance(PAR, (float, int)):
            raise e.TypeError("`PAR` should be a float or integer")
        if not 0 <= PAR <= 1:
            raise e.ValueError("`PAR` should be between 0 and 1")
        self._PAR = PAR

    @property
    def bw(self) -> float:
        return self._bw

    @bw.setter
    def bw(self, bw: float) -> None:
        if not isinstance(bw, (float, int)):
            raise e.TypeError("`bw` should be a float or integer")
        if bw < 0:
            raise e.ValueError("`bw` should be >= 0")
        self._bw = bw

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        n = pop.n_agents
        new_pos = torch.zeros(pop.n_variables, pop.n_dimensions, device=device)
        lb = pop.lb
        ub = pop.ub

        for j in range(pop.n_variables):
            r1 = torch.rand(1, device=device).item()
            if r1 < self.HMCR:
                # Memory consideration
                idx = torch.randint(0, n, (1,), device=device).item()
                new_pos[j] = pop.positions[idx, j]
                r2 = torch.rand(1, device=device).item()
                if r2 < self.PAR:
                    r3 = torch.rand(1, device=device).item()
                    new_pos[j] = new_pos[j] + r3 * self.bw
            else:
                new_pos[j] = torch.rand(pop.n_dimensions, device=device) * (ub[j] - lb[j]) + lb[j]

        return new_pos

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device

        new_pos = self._generate_new_harmony(pop, device)
        lb = pop.lb
        ub = pop.ub
        new_pos = new_pos.clamp(min=lb, max=ub)

        new_fit = fn(new_pos.unsqueeze(0))[0]

        # Replace worst if better
        worst_idx = pop.fitness.argmax()
        if new_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = new_pos
            pop.fitness[worst_idx] = new_fit


class IHS(HS):
    """Improved Harmony Search with adaptive PAR and bandwidth."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.PAR_min = 0.0
        self.PAR_max = 1.0
        self.bw_min = 1.0
        self.bw_max = 10.0

        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        self.PAR = self.PAR_min + (self.PAR_max - self.PAR_min) / T * t
        import math
        self.bw = self.bw_max * math.exp(
            (math.log(self.bw_min / (self.bw_max + c.EPSILON)) / T) * t
        )

        super().update(ctx)


class GHS(IHS):
    """Global-Best Harmony Search."""

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        n = pop.n_agents
        new_pos = torch.zeros(pop.n_variables, pop.n_dimensions, device=device)
        lb = pop.lb
        ub = pop.ub

        for j in range(pop.n_variables):
            r1 = torch.rand(1, device=device).item()
            if r1 < self.HMCR:
                idx = torch.randint(0, n, (1,), device=device).item()
                new_pos[j] = pop.positions[idx, j]
                r2 = torch.rand(1, device=device).item()
                if r2 < self.PAR:
                    z = torch.randint(0, pop.n_variables, (1,), device=device).item()
                    new_pos[j] = pop.best_position[z]
            else:
                new_pos[j] = torch.rand(pop.n_dimensions, device=device) * (ub[j] - lb[j]) + lb[j]

        return new_pos


class SGHS(GHS):
    """Self-Adaptive Global-Best Harmony Search."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.LP = 100
        self.HMCRm = 0.98
        self.PARm = 0.9

        super().__init__(params)

    def compile(self, population) -> None:
        self.lp = 1
        self.HMCR_history = []
        self.PAR_history = []

    def update(self, ctx: UpdateContext) -> None:
        device = ctx.space.population.device
        t = ctx.iteration
        T = max(ctx.n_iterations, 1)

        # Sample HMCR and PAR from Gaussian
        hmcr = min(max(torch.randn(1, device=device).item() * 0.01 + self.HMCRm, 0), 1)
        par = min(max(torch.randn(1, device=device).item() * 0.05 + self.PARm, 0), 1)
        self.HMCR = hmcr
        self.PAR = par
        self.HMCR_history.append(hmcr)
        self.PAR_history.append(par)

        # Adaptive bandwidth
        if t < T / 2:
            self.bw = self.bw_max - (self.bw_max - self.bw_min) * (2 * t / T)
        else:
            self.bw = self.bw_min

        # Generate and evaluate harmony (use HS.update)
        HS.update(self, ctx)

        # Update means every LP iterations
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
    """Novel Global-Best Harmony Search."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.pm = 0.1

        super().__init__(params)

    def _generate_new_harmony(self, pop, device) -> torch.Tensor:
        lb = pop.lb
        ub = pop.ub

        worst_idx = pop.fitness.argmax()
        worst = pop.positions[worst_idx]

        new_pos_range = 2 * (pop.best_position - worst)
        new_pos_range = new_pos_range.clamp(min=lb, max=ub)

        r1 = torch.rand(pop.n_variables, pop.n_dimensions, device=device)
        new_pos = worst + r1 * (new_pos_range - worst)

        # Random mutation
        if torch.rand(1, device=device).item() < self.pm:
            j = torch.randint(0, pop.n_variables, (1,), device=device).item()
            new_pos[j] = torch.rand(pop.n_dimensions, device=device) * (ub[j] - lb[j]) + lb[j]

        return new_pos


class GOGHS(NGHS):
    """Generalized Opposition-based Global-Best Harmony Search."""

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        fn = ctx.function
        device = pop.device

        new_pos = self._generate_new_harmony(pop, device)
        lb = pop.lb
        ub = pop.ub
        new_pos = new_pos.clamp(min=lb, max=ub)
        new_fit = fn(new_pos.unsqueeze(0))[0]

        # Opposition-based harmony
        A = pop.positions.min(dim=0).values
        B = pop.positions.max(dim=0).values
        k = torch.rand(1, device=device)
        opp_pos = k * (A + B) - new_pos
        opp_pos = opp_pos.clamp(min=lb, max=ub)
        opp_fit = fn(opp_pos.unsqueeze(0))[0]

        # Keep better
        if opp_fit < new_fit:
            new_pos = opp_pos
            new_fit = opp_fit

        worst_idx = pop.fitness.argmax()
        if new_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = new_pos
            pop.fitness[worst_idx] = new_fit
