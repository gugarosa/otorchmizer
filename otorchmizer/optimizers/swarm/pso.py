"""Particle Swarm Optimization family — fully vectorized with PyTorch.

Includes: PSO, AIWPSO, RPSO, SAVPSO, VPSO
All update rules operate on full population tensors (no per-agent loops).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class PSO(Optimizer):
    """Particle Swarm Optimization.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi.
        Swarm intelligence. Artificial Intelligence (2001).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: Optimizer -> PSO.")

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")
        self._w = w

    @property
    def c1(self) -> float:
        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")
        self._c1 = c1

    @property
    def c2(self) -> float:
        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")
        self._c2 = c2

    def compile(self, population: Population) -> None:
        dev = population.device
        shape = (population.n_agents, population.n_variables, population.n_dimensions)

        self.local_position = torch.zeros(shape, device=dev)
        self.local_fitness = torch.full((population.n_agents,), c.FLOAT_MAX, device=dev)
        self.velocity = torch.zeros(shape, device=dev)

    def evaluate(self, population: Population, function: Function) -> None:
        """Custom evaluate with local-best tracking — vectorized."""

        new_fitness = function(population.positions)

        # Update local bests where improved (no loop)
        improved = new_fitness < self.local_fitness
        if improved.any():
            self.local_position[improved] = population.positions[improved].clone()
            self.local_fitness[improved] = new_fitness[improved].clone()

        population.fitness = self.local_fitness.clone()
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Vectorized PSO velocity + position update."""

        pop = ctx.space.population
        shape = pop.positions.shape

        r1 = torch.rand(shape, device=pop.device)
        r2 = torch.rand(shape, device=pop.device)

        best = pop.best_position.unsqueeze(0)  # (1, n_vars, n_dims)

        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        pop.positions = pop.positions + self.velocity


class AIWPSO(PSO):
    """Adaptive Inertia Weight PSO.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: PSO -> AIWPSO.")

        self.w_min = 0.1
        self.w_max = 0.9

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def w_min(self) -> float:
        return self._w_min

    @w_min.setter
    def w_min(self, w_min: float) -> None:
        if not isinstance(w_min, (float, int)):
            raise e.TypeError("`w_min` should be a float or integer")
        if w_min < 0:
            raise e.ValueError("`w_min` should be >= 0")
        self._w_min = w_min

    @property
    def w_max(self) -> float:
        return self._w_max

    @w_max.setter
    def w_max(self, w_max: float) -> None:
        if not isinstance(w_max, (float, int)):
            raise e.TypeError("`w_max` should be a float or integer")
        if w_max < 0:
            raise e.ValueError("`w_max` should be >= 0")
        if w_max < self.w_min:
            raise e.ValueError("`w_max` should be >= `w_min`")
        self._w_max = w_max

    def _compute_success(self, population: Population) -> None:
        """Updates inertia weight based on improvement ratio (eq. 16)."""

        improved = (population.fitness < self._prev_fitness).float()
        p = improved.mean()
        self.w = (self.w_max - self.w_min) * p.item() + self.w_min

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population

        if ctx.iteration == 0:
            self._prev_fitness = pop.fitness.clone()

        # Standard PSO update
        shape = pop.positions.shape
        r1 = torch.rand(shape, device=pop.device)
        r2 = torch.rand(shape, device=pop.device)

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

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        IEEE Congress on Evolutionary Computation (2020).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: PSO -> RPSO.")
        super().__init__(params)
        logger.info("Class overrided.")

    def compile(self, population: Population) -> None:
        super().compile(population)
        dev = population.device
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.mass = r.generate_uniform_random_number(size=shape, device=dev)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        shape = pop.positions.shape

        max_velocity = self.velocity.abs().max().clamp(min=c.EPSILON)
        gamma = 1.0 / torch.sqrt(1.0 - (max_velocity ** 2 / c.LIGHT_SPEED ** 2))

        r1 = torch.rand(shape, device=pop.device)
        r2 = torch.rand(shape, device=pop.device)

        best = pop.best_position.unsqueeze(0)

        self.velocity = (
            self.mass * self.velocity * gamma
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        pop.positions = pop.positions + self.velocity


class SAVPSO(PSO):
    """Self-Adaptive Velocity PSO.

    References:
        H. Lu and W. Chen.
        Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: PSO -> SAVPSO.")
        super().__init__(params)
        logger.info("Class overrided.")

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        shape = pop.positions.shape
        n = pop.n_agents

        # Mean position across population
        mean_pos = pop.positions.mean(dim=0, keepdim=True)  # (1, n_vars, n_dims)

        # Random partner for each agent
        idx = torch.randint(0, n, (n,), device=pop.device)

        r1 = torch.rand(shape, device=pop.device)

        self.velocity = (
            self.w * torch.abs(self.local_position[idx] - self.local_position) * torch.sign(self.velocity)
            + r1 * (self.local_position - pop.positions)
            + (1 - r1) * (pop.best_position.unsqueeze(0) - pop.positions)
        )

        new_pos = pop.positions + self.velocity

        # Boundary handling via mean position
        r4 = torch.rand(shape, device=pop.device)
        ub = pop.ub.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        over = new_pos > ub
        under = new_pos < lb
        new_pos = torch.where(over, mean_pos + r4 * (ub - mean_pos), new_pos)
        new_pos = torch.where(under, mean_pos + r4 * (lb - mean_pos), new_pos)

        pop.positions = new_pos


class VPSO(PSO):
    """Vertical PSO.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application.
        International Conference on Machine Learning and Cybernetics (2007).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        logger.info("Overriding class: PSO -> VPSO.")
        super().__init__(params)
        logger.info("Class overrided.")

    def compile(self, population: Population) -> None:
        super().compile(population)
        dev = population.device
        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.v_velocity = torch.ones(shape, device=dev)

    def update(self, ctx: UpdateContext) -> None:
        pop = ctx.space.population
        shape = pop.positions.shape

        r1 = torch.rand(shape, device=pop.device)
        r2 = torch.rand(shape, device=pop.device)

        best = pop.best_position.unsqueeze(0)

        # Horizontal velocity (eq. 3)
        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.local_position - pop.positions)
            + self.c2 * r2 * (best - pop.positions)
        )

        # Vertical velocity (eq. 4)
        # Projection of v_velocity onto velocity direction
        vel_flat = self.velocity.reshape(pop.n_agents, -1)
        vv_flat = self.v_velocity.reshape(pop.n_agents, -1)

        dot_vv = (vel_flat * vv_flat).sum(dim=1, keepdim=True)  # (n, 1)
        dot_vv_norm = (vel_flat * vel_flat).sum(dim=1, keepdim=True) + c.EPSILON

        proj = (dot_vv / dot_vv_norm) * vel_flat  # (n, d)
        self.v_velocity = (vv_flat - proj).reshape(shape)

        # Position update (eq. 5)
        r1 = torch.rand(shape, device=pop.device)
        pop.positions = pop.positions + r1 * self.velocity + (1 - r1) * self.v_velocity
