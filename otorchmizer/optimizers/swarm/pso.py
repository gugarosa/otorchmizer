# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Particle Swarm Optimization family — fully vectorized with PyTorch.

Includes: PSO, AIWPSO, RPSO, SAVPSO, VPSO
All update rules operate on full population tensors (no per-agent loops).

"""

from __future__ import annotations

import math
from typing import Any

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population


class PSO(Optimizer):
    """Particle Swarm Optimization.

    Notes:
        Based on J. Kennedy, R. C. Eberhart, and Y. Shi, "Swarm Intelligence,"
        Artificial Intelligence (2001).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        super().__init__(params)

    @property
    def w(self) -> float:
        """Return the inertia weight."""

        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise TypeError("`w` must be a float or integer.")
        if not math.isfinite(w) or w < 0:
            raise ValueError("`w` must be finite and non-negative.")
        self._w = w

    @property
    def c1(self) -> float:
        """Return the cognitive coefficient."""

        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise TypeError("`c1` must be a float or integer.")
        if not math.isfinite(c1) or c1 < 0:
            raise ValueError("`c1` must be finite and non-negative.")
        self._c1 = c1

    @property
    def c2(self) -> float:
        """Return the social coefficient."""

        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise TypeError("`c2` must be a float or integer.")
        if not math.isfinite(c2) or c2 < 0:
            raise ValueError("`c2` must be finite and non-negative.")
        self._c2 = c2

    def compile(self, population: Population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        dev = population.device
        dt = population.dtype
        shape = (population.n_agents, population.n_variables, population.n_dimensions)

        self.local_position = torch.zeros(shape, device=dev, dtype=dt)
        self.local_fitness = torch.full((population.n_agents,), torch.inf, device=dev, dtype=dt)
        self.velocity = torch.zeros(shape, device=dev, dtype=dt)

    def evaluate(self, population: Population, function: Function) -> None:
        """Evaluate the population and update stored best solutions.

        Args:
            population: Population to evaluate.
            function: Objective function applied to the population.

        """

        new_fitness = function(population.positions)

        improved = new_fitness < self.local_fitness
        if improved.any():
            self.local_position[improved] = population.positions[improved].clone()
            self.local_fitness[improved] = new_fitness[improved].clone()

        population.fitness = new_fitness
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        shape = pop.positions.shape

        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        r2 = torch.rand(shape, device=pop.device, dtype=pop.dtype)

        best = pop.best_position.unsqueeze(0)

        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        pop.positions = pop.positions + self.velocity


class AIWPSO(PSO):
    """Adaptive Inertia Weight PSO.

    Notes:
        Based on A. Nickabadi, M. M. Ebadzadeh, and R. Safabakhsh,
        "A Novel Particle Swarm Optimization Algorithm with Adaptive Inertia Weight,"
        Applied Soft Computing (2011).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self._w_min = 0.1
        self._w_max = 0.9

        super().__init__(params)

    @property
    def w_min(self) -> float:
        """Return the minimum inertia weight."""

        return self._w_min

    @w_min.setter
    def w_min(self, w_min: float) -> None:
        self._validate_limits(w_min, self.w_max)
        self._w_min = w_min

    @property
    def w_max(self) -> float:
        """Return the maximum inertia weight."""

        return self._w_max

    @w_max.setter
    def w_max(self, w_max: float) -> None:
        self._validate_limits(self.w_min, w_max)
        self._w_max = w_max

    @staticmethod
    def _validate_limits(w_min: float, w_max: float) -> None:
        for name, value in (("w_min", w_min), ("w_max", w_max)):
            if not isinstance(value, (float, int)):
                raise TypeError(f"`{name}` must be a float or integer.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if w_max < w_min:
            raise ValueError("`w_max` must be greater than or equal to `w_min`.")

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Apply overrides without transiently invalid inertia limits.

        Args:
            params: Attribute overrides applied to the optimizer.

        """

        supplied = dict(params or {})
        remaining = dict(supplied)
        w_min = remaining.pop("w_min", self.w_min)
        w_max = remaining.pop("w_max", self.w_max)
        self._validate_limits(w_min, w_max)

        super().build(remaining)
        self._w_min, self._w_max = w_min, w_max
        self.params.update({name: value for name, value in (("w_min", w_min), ("w_max", w_max)) if name in supplied})

    def _compute_success(self, population: Population) -> None:

        improved = (population.fitness < self._prev_fitness).float()
        p = improved.mean()
        self.w = (self.w_max - self.w_min) * p.item() + self.w_min

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population

        if ctx.iteration == 0:
            self._prev_fitness = pop.fitness.clone()

        shape = pop.positions.shape
        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        r2 = torch.rand(shape, device=pop.device, dtype=pop.dtype)

        best = pop.best_position.unsqueeze(0)

        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        pop.positions = pop.positions + self.velocity

        self._compute_success(pop)
        self._prev_fitness = pop.fitness.clone()


class RPSO(PSO):
    """Relativistic PSO.

    Notes:
        Based on M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi, and J. P. Papa,
        "Harnessing Particle Swarm Optimization Through Relativistic Velocity,"
        IEEE Congress on Evolutionary Computation (2020).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population: Population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        super().compile(population)
        dev = population.device
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.mass = r.generate_uniform_random_number(size=shape, device=dev).to(dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        shape = pop.positions.shape

        max_velocity = self.velocity.abs().max().clamp(min=c.EPSILON)
        gamma = 1.0 / torch.sqrt(1.0 - (max_velocity**2 / c.LIGHT_SPEED**2))

        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        r2 = torch.rand(shape, device=pop.device, dtype=pop.dtype)

        best = pop.best_position.unsqueeze(0)

        self.velocity = (
            self.mass * self.velocity * gamma
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        pop.positions = pop.positions + self.velocity


class SAVPSO(PSO):
    """Self-Adaptive Velocity PSO.

    Notes:
        Based on H. Lu and W. Chen, "Self-Adaptive Velocity Particle Swarm Optimization
        for Solving Constrained Optimization Problems," Journal of Global Optimization (2008).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        shape = pop.positions.shape
        n = pop.n_agents

        mean_pos = pop.positions.mean(dim=0, keepdim=True)

        idx = torch.randint(0, n, (n,), device=pop.device)

        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)

        self.velocity = (
            self.w * torch.abs(self.local_position[idx] - self.local_position) * torch.sign(self.velocity)
            + r1 * (self.local_position - pop.positions)
            + (1 - r1) * (pop.best_position.unsqueeze(0) - pop.positions)
        )

        new_pos = pop.positions + self.velocity

        r4 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        ub = pop.ub.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        over = new_pos > ub
        under = new_pos < lb
        new_pos = torch.where(over, mean_pos + r4 * (ub - mean_pos), new_pos)
        new_pos = torch.where(under, mean_pos + r4 * (lb - mean_pos), new_pos)

        pop.positions = new_pos


class VPSO(PSO):
    """Vertical PSO.

    Notes:
        Based on W.-P. Yang, "Vertical Particle Swarm Optimization Algorithm and Its Application,"
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population: Population) -> None:
        """Initialize persistent optimizer state.

        Args:
            population: Population that defines the state shape, device, and dtype.

        """

        super().compile(population)
        dev = population.device
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.v_velocity = torch.ones(shape, device=dev, dtype=population.dtype)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one optimization step.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        shape = pop.positions.shape

        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        r2 = torch.rand(shape, device=pop.device, dtype=pop.dtype)

        best = pop.best_position.unsqueeze(0)

        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        vel_flat = self.velocity.reshape(pop.n_agents, -1)
        vv_flat = self.v_velocity.reshape(pop.n_agents, -1)

        dot_vv = (vel_flat * vv_flat).sum(dim=1, keepdim=True)
        dot_vv_norm = (vel_flat * vel_flat).sum(dim=1, keepdim=True).clamp_min(torch.finfo(pop.dtype).tiny)

        proj = (dot_vv / dot_vv_norm) * vel_flat
        self.v_velocity = (vv_flat - proj).reshape(shape)

        r1 = torch.rand(shape, device=pop.device, dtype=pop.dtype)
        pop.positions = pop.positions + r1 * self.velocity + (1 - r1) * self.v_velocity
