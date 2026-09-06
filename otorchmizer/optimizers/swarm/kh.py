# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Krill Herd.

References:
    A. H. Gandomi and A. H. Alavi.
    Krill herd: A new bio-inspired optimization algorithm.
    Communications in Nonlinear Science and Numerical Simulation (2012).

"""

from __future__ import annotations

import math
from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class KH(Optimizer):
    """Krill Herd optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.N_max = 0.01
        self.w_n = 0.42
        self.NN = 5
        self.V_f = 0.02
        self.w_f = 0.38
        self.D_max = 0.002
        self.C_t = 0.5
        self.Cr = 0.2
        self.Mu = 0.05

        super().__init__(params)

    @property
    def N_max(self) -> float:
        """Return the maximum induced speed."""

        return self._N_max

    @N_max.setter
    def N_max(self, value: float) -> None:
        self._N_max = self._validate_nonnegative("N_max", value)

    @property
    def w_n(self) -> float:
        """Return the induced-motion inertia weight."""

        return self._w_n

    @w_n.setter
    def w_n(self, value: float) -> None:
        self._w_n = self._validate_probability("w_n", value)

    @property
    def NN(self) -> int:
        """Return the sensing-distance divisor."""

        return self._NN

    @NN.setter
    def NN(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("`NN` must be an integer.")
        if value < 0:
            raise ValueError("`NN` must be non-negative.")
        self._NN = value

    @property
    def V_f(self) -> float:
        """Return the foraging speed."""

        return self._V_f

    @V_f.setter
    def V_f(self, value: float) -> None:
        self._V_f = self._validate_nonnegative("V_f", value)

    @property
    def w_f(self) -> float:
        """Return the foraging inertia weight."""

        return self._w_f

    @w_f.setter
    def w_f(self, value: float) -> None:
        self._w_f = self._validate_probability("w_f", value)

    @property
    def D_max(self) -> float:
        """Return the maximum diffusion speed."""

        return self._D_max

    @D_max.setter
    def D_max(self, value: float) -> None:
        self._D_max = self._validate_nonnegative("D_max", value)

    @property
    def C_t(self) -> float:
        """Return the position-update coefficient."""

        return self._C_t

    @C_t.setter
    def C_t(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("`C_t` must be a float or integer.")
        if not 0 <= value <= 2:
            raise ValueError("`C_t` must be between 0 and 2.")
        self._C_t = value

    @property
    def Cr(self) -> float:
        """Return the crossover scale."""

        return self._Cr

    @Cr.setter
    def Cr(self, value: float) -> None:
        self._Cr = self._validate_probability("Cr", value)

    @property
    def Mu(self) -> float:
        """Return the mutation scale."""

        return self._Mu

    @Mu.setter
    def Mu(self, value: float) -> None:
        self._Mu = self._validate_probability("Mu", value)

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(f"`{name}` must be a float or integer.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"`{name}` must be finite and non-negative.")
        return value

    @classmethod
    def _validate_probability(cls, name: str, value: float) -> float:
        value = cls._validate_nonnegative(name, value)
        if value > 1:
            raise ValueError(f"`{name}` must be between 0 and 1.")
        return value

    def compile(self, population) -> None:
        """Initialize persistent induced and foraging motion.

        Args:
            population: Population that defines state shape, device, and dtype.

        """

        self.motion = torch.zeros_like(population.positions)
        self.foraging = torch.zeros_like(population.positions)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one motion and genetic-operator cycle.

        Args:
            ctx: Population, objective function, and iteration state.

        """

        pop = ctx.space.population
        if not torch.isfinite(pop.fitness).all():
            raise ValueError("`population.fitness` must contain only finite values.")
        order = torch.argsort(pop.fitness)
        pop.positions = pop.positions[order]
        pop.fitness = pop.fitness[order]
        self.motion = self.motion[order]
        self.foraging = self.foraging[order]
        positions = pop.positions
        fitness = pop.fitness
        n = pop.n_agents
        tiny = torch.finfo(pop.dtype).tiny
        epsilon = torch.finfo(pop.dtype).eps
        fitness_scale = fitness.abs().max()
        scaled_fitness = torch.where(
            fitness_scale > 0,
            fitness / fitness_scale,
            fitness,
        )
        fitness_range = scaled_fitness[-1] - scaled_fitness[0]
        tied_fitness = fitness_range.abs() <= epsilon
        normalized_fitness = torch.where(
            tied_fitness,
            torch.zeros_like(scaled_fitness),
            (scaled_fitness - scaled_fitness[0]) / fitness_range,
        )
        distances = torch.cdist(positions.reshape(n, -1), positions.reshape(n, -1))
        safe_distances = distances.clamp_min(tiny)

        if self.NN:
            sensing_distance = distances.sum(dim=1) / (self.NN * n)
            neighbours = (distances < sensing_distance[:, None]) & ~torch.eye(
                n,
                device=pop.device,
                dtype=torch.bool,
            )
            local_fitness = torch.where(
                tied_fitness,
                torch.zeros((n, n), device=pop.device, dtype=pop.dtype),
                (scaled_fitness[:, None] - scaled_fitness[None, :]) / fitness_range,
            )
            direction = (positions.unsqueeze(0) - positions.unsqueeze(1)) / safe_distances[:, :, None, None]
            local_alpha = (local_fitness[:, :, None, None] * direction * neighbours[:, :, None, None]).sum(dim=1)
        else:
            local_alpha = torch.zeros_like(positions)

        best = positions[:1]
        distance_to_best = torch.linalg.vector_norm(
            (positions - best).reshape(n, -1),
            dim=1,
        ).clamp_min(tiny)
        best_effect = 2 * (torch.rand(n, device=pop.device, dtype=pop.dtype) + ctx.iteration / max(ctx.n_iterations, 1))
        target_alpha = (
            best_effect[:, None, None]
            * normalized_fitness[:, None, None]
            * (best - positions)
            / distance_to_best[:, None, None]
        )
        self.motion = self.N_max * (local_alpha + target_alpha) + self.w_n * self.motion

        if fitness.abs().max() < tiny:
            food_position = positions.mean(dim=0, keepdim=True)
        else:
            safe_fitness = torch.where(
                fitness.abs() < tiny,
                torch.full_like(fitness, tiny),
                fitness,
            )
            food_weights = safe_fitness.reciprocal()
            food_weights = food_weights / food_weights.abs().max()
            weight_sum = food_weights.sum()
            if weight_sum.abs() < tiny:
                food_position = positions.mean(dim=0, keepdim=True)
            else:
                food_position = (food_weights[:, None, None] * positions).sum(dim=0, keepdim=True) / weight_sum
        food_position = food_position.clamp(min=pop.lb.unsqueeze(0), max=pop.ub.unsqueeze(0))
        food_fitness = ctx.function(food_position)[0]
        self._update_archive(pop, food_position, food_fitness.unsqueeze(0))
        distance_to_food = torch.linalg.vector_norm(
            (positions - food_position).reshape(n, -1),
            dim=1,
        ).clamp_min(tiny)
        food_effect = 2 * (1 - ctx.iteration / max(ctx.n_iterations, 1))
        comparison_scale = torch.maximum(fitness_scale, food_fitness.abs())
        scaled_food = torch.where(
            comparison_scale > 0,
            food_fitness / comparison_scale,
            food_fitness,
        )
        scaled_agents = torch.where(
            comparison_scale > 0,
            fitness / comparison_scale,
            fitness,
        )
        food_difference = scaled_agents - scaled_food
        scaled_range = scaled_agents[-1] - scaled_agents[0]
        fallback_scale = food_difference.abs().max()
        food_fitness_factor = torch.where(
            scaled_range.abs() > epsilon,
            food_difference / scaled_range,
            torch.where(
                fallback_scale > epsilon,
                food_difference / fallback_scale,
                torch.zeros_like(food_difference),
            ),
        )
        food_beta = (
            food_effect
            * food_fitness_factor[:, None, None]
            * (food_position - positions)
            / distance_to_food[:, None, None]
        )
        best_beta = normalized_fitness[:, None, None] * (best - positions) / distance_to_best[:, None, None]
        self.foraging = self.V_f * (food_beta + best_beta) + self.w_f * self.foraging

        progress = ctx.iteration / max(ctx.n_iterations, 1)
        diffusion = self.D_max * (1 - progress) * (torch.rand_like(positions) * 2 - 1)
        delta_t = self.C_t * (pop.ub - pop.lb).sum()
        candidates = positions + delta_t * (self.motion + self.foraging + diffusion)

        if n > 1:
            partner = torch.randint(0, n - 1, (n,), device=pop.device)
            partner += (partner >= torch.arange(n, device=pop.device)).long()
            crossover_probability = self.Cr * normalized_fitness
            crossover = torch.rand_like(candidates) < crossover_probability[:, None, None]
            candidates = torch.where(crossover, candidates[partner], candidates)

            first = torch.randint(0, n - 1, (n,), device=pop.device)
            second = torch.randint(0, n - 1, (n,), device=pop.device)
            agent_index = torch.arange(n, device=pop.device)
            first += (first >= agent_index).long()
            second += (second >= agent_index).long()
            mutation_probability = (self.Mu / normalized_fitness.clamp_min(tiny)).clamp(max=1)
            mutation = torch.rand_like(candidates) < mutation_probability[:, None, None]
            differential = best + torch.rand_like(candidates) * (positions[first] - positions[second])
            candidates = torch.where(mutation, differential, candidates)

        pop.positions = candidates.clamp(
            min=pop.lb.unsqueeze(0),
            max=pop.ub.unsqueeze(0),
        )

    @staticmethod
    def _update_archive(population, positions: torch.Tensor, fitness: torch.Tensor) -> None:
        best_idx = fitness.argmin()
        if fitness[best_idx] < population.best_fitness:
            population.best_fitness = fitness[best_idx].clone()
            population.best_position = positions[best_idx].clone()
