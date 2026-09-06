# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Water Wave Optimization.

References:
    Y.-J. Zheng.
    Water wave optimization: A new nature-inspired metaheuristic.
    Computers & Operations Research (2015).
"""

from __future__ import annotations

from typing import Any

import torch

from otorchmizer.core.optimizer import Optimizer, UpdateContext


class WWO(Optimizer):
    """Water Wave Optimization with propagation, breaking, and refraction phases.

    Notes:
        Fitness values must be finite and non-negative because wavelength adaptation uses objective ratios.
        Exact-zero ratios preserve the current wavelength, while equal populations apply one standard reduction step.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the WWO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.h_max = 5
        self.alpha = 1.001
        self.beta = 0.001
        self.k_max = 1
        super().__init__(params)

    @property
    def h_max(self) -> int:
        """Return the maximum wave height.

        Returns:
            int: Current maximum wave height.

        """

        return self._h_max

    @h_max.setter
    def h_max(self, value: int) -> None:
        """Set the maximum wave height.

        Args:
            value: New maximum wave height.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`h_max` must be an integer.")
        if value <= 0:
            raise ValueError("`h_max` must be positive.")
        self._h_max = value

    @property
    def alpha(self) -> float:
        """Return the wavelength-reduction coefficient.

        Returns:
            float: Current wavelength-reduction coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the wavelength-reduction coefficient.

        Args:
            value: New wavelength-reduction coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, (float, int)):
            raise TypeError("`alpha` must be a float or integer.")
        if value <= 0:
            raise ValueError("`alpha` must be positive.")
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        """Return the breaking coefficient.

        Returns:
            float: Current breaking coefficient.

        """

        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the breaking coefficient.

        Args:
            value: New breaking coefficient.

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
    def k_max(self) -> int:
        """Return the maximum number of breaking trials.

        Returns:
            int: Current maximum breaking count.

        """

        return self._k_max

    @k_max.setter
    def k_max(self, value: int) -> None:
        """Set the maximum number of breaking trials.

        Args:
            value: New maximum breaking count.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise TypeError("`k_max` must be an integer.")
        if value <= 0:
            raise ValueError("`k_max` must be positive.")
        self._k_max = value

    def compile(self, population) -> None:
        """Initialize wave height and wavelength state.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.height = torch.full(
            (population.n_agents,),
            self.h_max,
            dtype=torch.long,
            device=population.device,
        )
        self.length = population.fitness.new_full((population.n_agents,), 0.5)

    @staticmethod
    def _validate_fitness(fitness: torch.Tensor, offender: str) -> None:
        if not torch.isfinite(fitness).all() or (fitness < 0).any():
            raise ValueError(f"`{offender}` must contain finite non-negative values for WWO.")

    @staticmethod
    def _repair(position: torch.Tensor, population) -> torch.Tensor:
        outside = (position < population.lb) | (position > population.ub)
        if not outside.any():
            return position
        replacement = torch.rand_like(position) * (population.ub - population.lb) + population.lb
        return torch.where(outside, replacement, position)

    @staticmethod
    def _scaled_ratio(length: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if (
            not torch.isfinite(length)
            or not torch.isfinite(numerator)
            or not torch.isfinite(denominator)
            or length <= 0
            or numerator <= 0
            or denominator <= 0
        ):
            raise ValueError("`wavelength` ratio operands must be finite and positive.")

        length_mantissa, length_exponent = torch.frexp(length)
        numerator_mantissa, numerator_exponent = torch.frexp(numerator)
        denominator_mantissa, denominator_exponent = torch.frexp(denominator)
        mantissa = length_mantissa * numerator_mantissa / denominator_mantissa
        exponent = length_exponent + numerator_exponent - denominator_exponent
        mantissa, mantissa_exponent = torch.frexp(mantissa)
        exponent = exponent + mantissa_exponent

        result = mantissa
        remaining = exponent
        # Torch 2.0 overflows ldexp at magnitude 128, so compose all IEEE ranges in safe chunks
        for _ in range(32):
            step = remaining.clamp(min=-126, max=126)
            result = torch.ldexp(result, step)
            remaining = remaining - step
        result = result.to(dtype=length.dtype)
        if not torch.isfinite(result) or result <= 0:
            raise ValueError("`wavelength` update must be representable in the population dtype.")
        return result

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population through propagation, breaking, and refraction.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        self._validate_fitness(pop.fitness, "population.fitness")
        self._validate_fitness(pop.best_fitness.reshape(1), "population.best_fitness")
        width = pop.ub - pop.lb

        for i in range(pop.n_agents):
            direction = torch.rand(pop.n_variables, 1, device=pop.device, dtype=pop.dtype) * 2 - 1
            propagated = pop.positions[i] + direction * self.length[i] * width
            propagated = self._repair(propagated, pop)
            propagated_fit = fn(propagated.unsqueeze(0))[0]
            self._validate_fitness(propagated_fit.reshape(1), "function")

            if propagated_fit < pop.fitness[i]:
                if propagated_fit < pop.best_fitness:
                    pop.best_position = propagated.clone()
                    pop.best_fitness = propagated_fit.clone()

                    dimensions = torch.randperm(pop.n_variables, device=pop.device)[: min(self.k_max, pop.n_variables)]
                    for j in dimensions:
                        broken = propagated.clone()
                        broken[j] += torch.randn_like(broken[j]) * self.beta * width[j]
                        broken = self._repair(broken, pop)
                        broken_fit = fn(broken.unsqueeze(0))[0]
                        self._validate_fitness(broken_fit.reshape(1), "function")
                        if broken_fit < propagated_fit:
                            if propagated_fit > 0 and broken_fit > 0:
                                self.length[i] = self._scaled_ratio(self.length[i], broken_fit, propagated_fit)
                            propagated = broken
                            propagated_fit = broken_fit
                            pop.best_position = broken.clone()
                            pop.best_fitness = broken_fit.clone()

                pop.positions[i] = propagated
                pop.fitness[i] = propagated_fit
                self.height[i] = self.h_max
                continue

            self.height[i] -= 1
            if self.height[i] > 0:
                continue

            old_fitness = pop.fitness[i].clone()
            mean = (pop.best_position + pop.positions[i]) / 2
            std = (pop.best_position - pop.positions[i]).abs() / 2
            refracted = mean + torch.randn_like(mean) * std
            refracted = self._repair(refracted, pop)
            refracted_fit = fn(refracted.unsqueeze(0))[0]
            self._validate_fitness(refracted_fit.reshape(1), "function")
            pop.positions[i] = refracted
            pop.fitness[i] = refracted_fit
            self.height[i] = self.h_max
            if old_fitness > 0 and refracted_fit > 0:
                self.length[i] = self._scaled_ratio(self.length[i], old_fitness, refracted_fit)

        best_fit = pop.fitness.min()
        worst_fit = pop.fitness.max()
        spread = worst_fit - best_fit
        exponent = -torch.ones_like(pop.fitness) if spread == 0 else -(worst_fit - pop.fitness) / spread
        self.length *= self.alpha**exponent
        pop.update_best()
