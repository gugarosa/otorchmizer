# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Remaining science-based optimizers: AIG, CDO, EFO, ESA, HGSO, LSA, MOA, SMA, TEO, TWO, WEO."""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


def _nonzero(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


class AIG(Optimizer):
    """Algorithm of the Innovative Gunner."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the AIG optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 3.14159
        self.beta = 3.14159
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one AIG step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)

        t = ctx.iteration / max(ctx.n_iterations, 1)
        alpha_max = self.alpha * (1 - t)
        beta_max = self.beta * (1 - t)

        for i in range(n):
            alpha_corr = torch.randn(1, device=device) * alpha_max
            beta_corr = torch.randn(1, device=device) * beta_max

            g_alpha = torch.cos(alpha_corr) if alpha_corr < 0 else 1.0 / torch.cos(alpha_corr).clamp(min=0.01)
            g_beta = torch.cos(beta_corr) if beta_corr < 0 else 1.0 / torch.cos(beta_corr).clamp(min=0.01)

            new_pos = pop.positions[i] * g_alpha * g_beta
            new_pos = new_pos.clamp(min=lb.squeeze(0), max=ub.squeeze(0))
            new_fit = fn(new_pos.unsqueeze(0))[0]
            if new_fit < pop.fitness[i]:
                pop.positions[i] = new_pos
                pop.fitness[i] = new_fit


class CDO(Optimizer):
    """Chernobyl Disaster Optimizer."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the CDO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if population.n_agents < 3:
            raise e.SizeError("`population.n_agents` must be at least 3 for CDO.")

        shape = (population.n_variables, population.n_dimensions)
        self.gamma_pos = population.positions.new_zeros(shape)
        self.beta_pos = population.positions.new_zeros(shape)
        self.alpha_pos = population.positions.new_zeros(shape)
        self.gamma_fit = population.fitness.new_full((), torch.inf)
        self.beta_fit = population.fitness.new_full((), torch.inf)
        self.alpha_fit = population.fitness.new_full((), torch.inf)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one CDO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        leaders = torch.argsort(pop.fitness)[:3]
        self.alpha_pos, self.alpha_fit = pop.positions[leaders[0]].clone(), pop.fitness[leaders[0]].clone()
        self.beta_pos, self.beta_fit = pop.positions[leaders[1]].clone(), pop.fitness[leaders[1]].clone()
        self.gamma_pos, self.gamma_fit = pop.positions[leaders[2]].clone(), pop.fitness[leaders[2]].clone()

        ws = 3 * (1 - t)
        sampling_dtype = torch.float64 if pop.dtype == torch.float64 else torch.float32
        one = torch.tensor(1.0, device=pop.device, dtype=sampling_dtype)
        s_gamma = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 299999).to(pop.dtype)
        s_beta = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 269999).to(pop.dtype)
        s_alpha = torch.log10(one + torch.rand((), device=pop.device, dtype=sampling_dtype) * 15999).to(pop.dtype)
        s_gamma, s_beta, s_alpha = _nonzero(s_gamma), _nonzero(s_beta), _nonzero(s_alpha)

        def component(target, source_scale, denominator):
            r1 = torch.rand_like(pop.positions)
            r2 = torch.rand_like(pop.positions)
            r3 = torch.rand_like(pop.positions)
            rho = torch.pi * r1.square() / denominator - ws * r2
            gradient = (torch.pi * r3.square() * target.unsqueeze(0) - pop.positions).abs()
            return source_scale * (pop.positions - rho * gradient)

        v_gamma = component(self.gamma_pos, 1.0, s_gamma)
        v_beta = component(self.beta_pos, 0.5, 0.5 * s_beta)
        v_alpha = component(self.alpha_pos, 0.25, 0.25 * s_alpha)
        pop.positions = ((v_alpha + v_beta + v_gamma) / 3).clamp(min=lb, max=ub)


class EFO(Optimizer):
    """Electromagnetic Field Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the EFO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.positive_field = 0.1
        self.negative_field = 0.5
        self.ps_ratio = 0.1
        self.r_ratio = 0.4
        self.phi = (1 + 5**0.5) / 2
        self.RI = 0
        super().__init__(params)

    @property
    def positive_field(self) -> float:
        """Return the positive-field proportion.

        Returns:
            float: Current positive-field proportion.

        """

        return self._positive_field

    @positive_field.setter
    def positive_field(self, value: float) -> None:
        """Set the positive-field proportion.

        Args:
            value: New positive-field proportion.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`positive_field` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`positive_field` must be between 0 and 1.")
        self._positive_field = float(value)

    @property
    def negative_field(self) -> float:
        """Return the negative-field proportion.

        Returns:
            float: Current negative-field proportion.

        """

        return self._negative_field

    @negative_field.setter
    def negative_field(self, value: float) -> None:
        """Set the negative-field proportion.

        Args:
            value: New negative-field proportion.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`negative_field` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`negative_field` must be between 0 and 1.")
        self._negative_field = float(value)

    @property
    def ps_ratio(self) -> float:
        """Return the positive-selection probability.

        Returns:
            float: Current positive-selection probability.

        """

        return self._ps_ratio

    @ps_ratio.setter
    def ps_ratio(self, value: float) -> None:
        """Set the positive-selection probability.

        Args:
            value: New positive-selection probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`ps_ratio` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`ps_ratio` must be between 0 and 1.")
        self._ps_ratio = float(value)

    @property
    def r_ratio(self) -> float:
        """Return the random-reset probability.

        Returns:
            float: Current random-reset probability.

        """

        return self._r_ratio

    @r_ratio.setter
    def r_ratio(self, value: float) -> None:
        """Set the random-reset probability.

        Args:
            value: New random-reset probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`r_ratio` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`r_ratio` must be between 0 and 1.")
        self._r_ratio = float(value)

    @property
    def phi(self) -> float:
        """Return the golden-ratio coefficient.

        Returns:
            float: Current golden-ratio coefficient.

        """

        return self._phi

    @phi.setter
    def phi(self, value: float) -> None:
        """Set the golden-ratio coefficient.

        Args:
            value: New golden-ratio coefficient.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`phi` must be a float or integer.")
        self._phi = float(value)

    @property
    def RI(self) -> int:
        """Return the rotating reset index.

        Returns:
            int: Current reset index.

        """

        return self._RI

    @RI.setter
    def RI(self, value: int) -> None:
        """Set the rotating reset index.

        Args:
            value: New reset index.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is negative.

        """

        if not isinstance(value, int):
            raise e.TypeError("`RI` must be an integer.")
        if value < 0:
            raise e.ValueError("`RI` must be non-negative.")
        self._RI = value

    def compile(self, population) -> None:
        """Validate the population required by the electromagnetic fields.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            SizeError: If fewer than three agents are available.

        """

        if population.n_agents < 3:
            raise e.SizeError("`population.n_agents` must be at least 3 for EFO.")
        positive_end = max(int(population.n_agents * self.positive_field), 1)
        negative_start = min(
            max(int(population.n_agents * (1 - self.negative_field)), positive_end + 1),
            population.n_agents - 1,
        )
        if positive_end >= negative_start:
            raise e.ValueError("`positive_field` and `negative_field` must leave a non-empty neutral field.")
        if self.RI >= population.n_variables:
            raise e.ValueError("`RI` must be smaller than `population.n_variables`.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one EFO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        sorted_idx = torch.argsort(pop.fitness)
        positive_end = max(int(n * self.positive_field), 1)
        negative_start = min(max(int(n * (1 - self.negative_field)), positive_end + 1), n - 1)
        force = torch.rand((), device=device, dtype=pop.dtype)
        candidate = pop.positions[sorted_idx[0]].clone()

        for j in range(pop.n_variables):
            pos_rank = torch.randint(0, positive_end, (), device=device).item()
            neg_rank = torch.randint(negative_start, n, (), device=device).item()
            neutral_rank = torch.randint(positive_end, negative_start, (), device=device).item()
            pos = pop.positions[sorted_idx[pos_rank], j]
            neg = pop.positions[sorted_idx[neg_rank], j]
            neutral = pop.positions[sorted_idx[neutral_rank], j]

            if torch.rand((), device=device) < self.ps_ratio:
                candidate[j] = pos
            else:
                candidate[j] = neg + self.phi * force * (pos - neutral) - force * (neg - neutral)

        if torch.rand((), device=device) < self.r_ratio:
            candidate[self.RI] = (
                torch.rand_like(candidate[self.RI]) * (pop.ub[self.RI] - pop.lb[self.RI]) + pop.lb[self.RI]
            )
            self.RI = (self.RI + 1) % pop.n_variables

        candidate = candidate.clamp(min=pop.lb, max=pop.ub)
        candidate_fit = fn(candidate.unsqueeze(0))[0]
        worst_idx = sorted_idx[-1]
        if candidate_fit < pop.fitness[worst_idx]:
            pop.positions[worst_idx] = candidate
            pop.fitness[worst_idx] = candidate_fit
            pop.update_best()


class ESA(Optimizer):
    """Electro-Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the ESA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.n_electrons = 5
        super().__init__(params)

    @property
    def n_electrons(self) -> int:
        """Return the number of sampled electrons.

        Returns:
            int: Current electron count.

        """

        return self._n_electrons

    @n_electrons.setter
    def n_electrons(self, value: int) -> None:
        """Set the number of sampled electrons.

        Args:
            value: New electron count.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise e.TypeError("`n_electrons` must be an integer.")
        if value <= 0:
            raise e.ValueError("`n_electrons` must be positive.")
        self._n_electrons = value

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.D = torch.rand_like(population.positions)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one ESA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        eps = torch.finfo(pop.dtype).eps

        for i in range(n):
            reference_best = pop.best_position.clone()
            levels = torch.randint(2, 6, (self.n_electrons, 1, 1), device=pop.device)
            radius = _nonzero(self.D[i]).unsqueeze(0)
            electrons = (
                pop.positions[i].unsqueeze(0)
                + (torch.rand_like(radius.expand(self.n_electrons, -1, -1)) * 2 - 1)
                * (1 - levels.to(pop.dtype).square().reciprocal())
                / radius
            )
            electrons = electrons.clamp(min=pop.lb, max=pop.ub)
            electron_fitness = fn(electrons)
            best_electron_index = electron_fitness.argmin()
            best_electron = electrons[best_electron_index]
            best_electron_fitness = electron_fitness[best_electron_index]
            if best_electron_fitness < pop.best_fitness:
                pop.best_position = best_electron.clone()
                pop.best_fitness = best_electron_fitness.clone()

            rydberg = torch.rand((), device=pop.device, dtype=pop.dtype)
            acceleration = torch.rand((), device=pop.device, dtype=pop.dtype)
            best_inverse = reference_best.square().clamp_min(eps).reciprocal()
            current_inverse = pop.positions[i].square().clamp_min(eps).reciprocal()
            self.D[i] = best_electron - reference_best + rydberg * (best_inverse - current_inverse)

            candidate = (pop.positions[i] + acceleration * self.D[i]).clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]
            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit

        pop.update_best()


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
            raise e.TypeError("`n_clusters` must be an integer.")
        if value <= 0:
            raise e.ValueError("`n_clusters` must be positive.")
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
            raise e.TypeError("`l1` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`l1` must be non-negative.")
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
            raise e.TypeError("`l2` must be a float or integer.")
        if value <= 0:
            raise e.ValueError("`l2` must be positive.")
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
            raise e.TypeError("`l3` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`l3` must be non-negative.")
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
            raise e.TypeError("`alpha` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`alpha` must be non-negative.")
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
            raise e.TypeError("`beta` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`beta` must be non-negative.")
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
            raise e.TypeError("`K` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`K` must be non-negative.")
        self._K = float(value)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        if self.n_clusters > population.n_agents:
            raise e.SizeError("`n_clusters` must not exceed `population.n_agents`.")

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
            raise e.ValueError("`population.fitness` must contain finite non-negative values for HGSO.")

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
            raise e.ValueError("`function` must return finite non-negative values for HGSO.")

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
                raise e.ValueError("`function` must return finite non-negative values for HGSO.")

            pop.positions[worst] = replacement_positions
            pop.fitness[worst] = replacement_fitness
            pop.update_best()


class LSA(Optimizer):
    """Lightning Search Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the LSA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.max_time = 10
        self.E = 2.05
        self.p_fork = 0.01
        super().__init__(params)

    @property
    def max_time(self) -> int:
        """Return the maximum channel time.

        Returns:
            int: Current maximum channel time.

        """

        return self._max_time

    @max_time.setter
    def max_time(self, value: int) -> None:
        """Set the maximum channel time.

        Args:
            value: New maximum channel time.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise e.TypeError("`max_time` must be an integer.")
        if value <= 0:
            raise e.ValueError("`max_time` must be positive.")
        self._max_time = value

    @property
    def E(self) -> float:
        """Return the initial energy.

        Returns:
            float: Current initial energy.

        """

        return self._E

    @E.setter
    def E(self, value: float) -> None:
        """Set the initial energy.

        Args:
            value: New initial energy.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`E` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`E` must be non-negative.")
        self._E = float(value)

    @property
    def p_fork(self) -> float:
        """Return the forking probability.

        Returns:
            float: Current forking probability.

        """

        return self._p_fork

    @p_fork.setter
    def p_fork(self, value: float) -> None:
        """Set the forking probability.

        Args:
            value: New forking probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`p_fork` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`p_fork` must be between 0 and 1.")
        self._p_fork = float(value)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.time = 0
        random_direction = torch.rand_like(population.positions[0]) * 2 - 1
        self.direction = torch.where(
            random_direction < 0, -torch.ones_like(random_direction), torch.ones_like(random_direction)
        )

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one LSA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        t = ctx.iteration / max(ctx.n_iterations, 1)

        self.time += 1
        if self.time >= self.max_time:
            worst_idx = pop.fitness.argmax()
            pop.positions[worst_idx] = pop.best_position.clone()
            pop.fitness[worst_idx] = pop.best_fitness.clone()
            self.time = 0

        order = torch.argsort(pop.fitness)
        pop.positions = pop.positions[order]
        pop.fitness = pop.fitness[order]
        best_position = pop.positions[0].clone()

        for j in range(pop.n_variables):
            shake = best_position.clone()
            shake[j] += self.direction[j] * 0.005 * (pop.ub[j] - pop.lb[j])
            shake = shake.clamp(min=pop.lb, max=pop.ub)
            shake_fitness = fn(shake.unsqueeze(0))[0]
            if shake_fitness < pop.best_fitness:
                pop.best_position = shake.clone()
                pop.best_fitness = shake_fitness.clone()
            if shake_fitness > pop.fitness[0]:
                self.direction[j] *= -1

        energy = self.E - 2 * torch.exp(pop.positions.new_tensor(-5 * (1 - t)))
        for i in range(n):
            candidate = pop.positions[i].clone()
            distance = pop.positions[i] - best_position
            zero = distance == 0
            below = distance < 0
            exponential = -torch.log(torch.rand_like(distance).clamp_min(torch.finfo(pop.dtype).tiny)) * distance.abs()
            candidate = torch.where(zero, candidate + self.direction * torch.randn_like(candidate) * energy, candidate)
            candidate = torch.where(below & ~zero, candidate + exponential, candidate)
            candidate = torch.where(~below & ~zero, candidate - exponential, candidate)
            candidate = candidate.clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]

            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit
                if torch.rand((), device=pop.device) < self.p_fork:
                    fork = torch.rand_like(candidate) * (pop.ub - pop.lb) + pop.lb
                    fork_fit = fn(fork.unsqueeze(0))[0]
                    if fork_fit < pop.fitness[i]:
                        pop.positions[i] = fork
                        pop.fitness[i] = fork_fit

        pop.update_best()


class MOA(Optimizer):
    """Magnetic Optimization Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the MOA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.alpha = 1.0
        self.rho = 2.0
        super().__init__(params)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one MOA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents

        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()

        norm_fit = (pop.fitness - best_fit) / (worst_fit - best_fit + c.EPSILON)
        mass = self.alpha + self.rho * norm_fit

        for i in range(n):
            force = torch.zeros_like(pop.positions[i])
            neighbors = [max(i - 1, 0), min(i + 1, n - 1)]

            for j in neighbors:
                if j == i:
                    continue
                diff = pop.positions[j] - pop.positions[i]
                dist = torch.linalg.norm(diff.reshape(-1)).clamp(min=1e-10)
                force += norm_fit[j] * diff / dist

            vel = force / (mass[i] + 1e-10) * torch.rand(1, device=device)
            pop.positions[i] = pop.positions[i] + vel


class SMA(Optimizer):
    """Slime Mould Algorithm."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the SMA optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.z = 0.03
        super().__init__(params)

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        shape = (population.n_agents, population.n_variables, population.n_dimensions)
        self.weight = population.positions.new_ones(shape)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one SMA step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        device = pop.device
        n = pop.n_agents
        best = pop.best_position.unsqueeze(0)
        lb = pop.lb.unsqueeze(0)
        ub = pop.ub.unsqueeze(0)
        t = ctx.iteration / max(ctx.n_iterations, 1)

        sorted_idx = torch.argsort(pop.fitness)
        best_fit = pop.fitness[sorted_idx[0]]
        worst_fit = pop.fitness[sorted_idx[-1]]
        fit_range = (worst_fit - best_fit).clamp_min(torch.finfo(pop.fitness.dtype).eps)

        # Update weights
        for rank, idx in enumerate(sorted_idx):
            r = torch.rand_like(self.weight[idx])
            log_val = torch.log10((pop.fitness[idx] - best_fit) / fit_range + 1)
            if rank < n // 2:
                self.weight[idx] = 1 + r * log_val
            else:
                self.weight[idx] = 1 - r * log_val

        a_val = torch.atanh(torch.tensor(-(t + 1) / (max(ctx.n_iterations, 1) + 1) + 1, device=device)).clamp(max=5)
        b_val = 1 - (t + 1) / (max(ctx.n_iterations, 1) + 1)

        for i in range(n):
            r = torch.rand(1, device=device).item()
            if r < self.z:
                pop.positions[i] = torch.rand_like(pop.positions[i]) * (ub.squeeze(0) - lb.squeeze(0)) + lb.squeeze(0)
            else:
                p = torch.tanh(torch.abs(pop.fitness[i] - best_fit))
                vb = torch.rand_like(pop.positions[i]) * 2 * a_val - a_val
                vc = torch.rand_like(pop.positions[i]) * 2 * b_val - b_val

                if torch.rand(1, device=device).item() < p.item():
                    k = torch.randint(0, n, (1,), device=device).item()
                    l_idx = torch.randint(0, n, (1,), device=device).item()
                    pop.positions[i] = best.squeeze(0) + vb * self.weight[i] * (pop.positions[k] - pop.positions[l_idx])
                else:
                    pop.positions[i] = pop.positions[i] * vc

        pop.positions = pop.positions.clamp(min=lb, max=ub)


class TEO(Optimizer):
    """Thermal Exchange Optimization.

    Notes:
        Fitness values must be finite and non-negative because they define the heat-transfer coefficient.
        For odd populations, the median ranked object uses itself as its environment while the remaining objects
        are paired between the better and worse halves.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the TEO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.c1 = True
        self.c2 = True
        self.pro = 0.05
        self.n_TM = 4
        super().__init__(params)

    @property
    def c1(self) -> bool:
        """Return the random step-size switch.

        Returns:
            bool: Current step-size switch.

        """

        return self._c1

    @c1.setter
    def c1(self, value: bool) -> None:
        """Set the random step-size switch.

        Args:
            value: New step-size switch.

        Raises:
            TypeError: If the value is not Boolean.

        """

        if not isinstance(value, bool):
            raise e.TypeError("`c1` must be a bool.")
        self._c1 = value

    @property
    def c2(self) -> bool:
        """Return the randomness switch.

        Returns:
            bool: Current randomness switch.

        """

        return self._c2

    @c2.setter
    def c2(self, value: bool) -> None:
        """Set the randomness switch.

        Args:
            value: New randomness switch.

        Raises:
            TypeError: If the value is not Boolean.

        """

        if not isinstance(value, bool):
            raise e.TypeError("`c2` must be a bool.")
        self._c2 = value

    @property
    def pro(self) -> float:
        """Return the reset probability.

        Returns:
            float: Current reset probability.

        """

        return self._pro

    @pro.setter
    def pro(self, value: float) -> None:
        """Set the reset probability.

        Args:
            value: New reset probability.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the unit interval.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`pro` must be a float or integer.")
        if not 0 <= value <= 1:
            raise e.ValueError("`pro` must be between 0 and 1.")
        self._pro = float(value)

    @property
    def n_TM(self) -> int:
        """Return the thermal-memory capacity.

        Returns:
            int: Current thermal-memory capacity.

        """

        return self._n_TM

    @n_TM.setter
    def n_TM(self, value: int) -> None:
        """Set the thermal-memory capacity.

        Args:
            value: New memory capacity.

        Raises:
            TypeError: If the value is not an integer.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, int):
            raise e.TypeError("`n_TM` must be an integer.")
        if value <= 0:
            raise e.ValueError("`n_TM` must be positive.")
        self._n_TM = value

    def compile(self, population) -> None:
        """Initialize optimizer state for a population.

        Args:
            population: Population whose tensors define the optimizer state.

        """

        self.environment = population.positions.clone()
        self.TM = []

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one TEO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        t = ctx.iteration / max(ctx.n_iterations, 1)
        if not torch.isfinite(pop.fitness).all() or (pop.fitness < 0).any():
            raise e.ValueError("`population.fitness` must contain finite non-negative values for TEO.")

        order = torch.argsort(pop.fitness)
        positions = pop.positions[order].clone()
        fitness = pop.fitness[order].clone()
        memory = self.TM or [(positions[i].clone(), fitness[i].clone()) for i in range(min(self.n_TM, n))]

        r = torch.rand(n, 1, 1, device=device, dtype=pop.dtype)
        factor = float(self.c1) + float(self.c2) * (1 - t)
        pair = torch.arange(n, device=device)
        half = n // 2
        pair[:half] = torch.arange(n - half, n, device=device)
        pair[n - half :] = torch.arange(half, device=device)
        modified_environment = (1 - factor * r) * positions
        environment = modified_environment[pair]

        worst_fit = fitness[-1]
        beta = torch.zeros_like(fitness) if worst_fit == 0 else fitness / worst_fit
        candidates = environment + (positions - environment) * torch.exp(-beta.view(n, 1, 1) * t)
        for i in range(n):
            if torch.rand((), device=device) < self.pro:
                j = torch.randint(0, pop.n_variables, (), device=device).item()
                candidates[i, j] = torch.rand_like(candidates[i, j]) * (pop.ub[j] - pop.lb[j]) + pop.lb[j]

        candidates = candidates.clamp(min=pop.lb, max=pop.ub)
        candidate_fitness = fn(candidates)
        if not torch.isfinite(candidate_fitness).all() or (candidate_fitness < 0).any():
            raise e.ValueError("`function` must return finite non-negative values for TEO.")

        memory_positions = torch.stack([position for position, _ in memory])
        memory_fitness = torch.stack([value for _, value in memory])
        combined_positions = torch.cat((candidates, memory_positions))
        combined_fitness = torch.cat((candidate_fitness, memory_fitness))
        selected = torch.argsort(combined_fitness)[:n]
        memory_selected = torch.argsort(combined_fitness)[: min(self.n_TM, combined_fitness.numel())]

        self.environment = environment
        self.TM = [(combined_positions[index].clone(), combined_fitness[index].clone()) for index in memory_selected]
        pop.positions = combined_positions[selected]
        pop.fitness = combined_fitness[selected]
        pop.update_best()


class TWO(Optimizer):
    """Tug of War Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the TWO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.mu_s = 1.0
        self.mu_k = 1.0
        self.delta_t = 1.0
        self.alpha = 0.9
        self.beta = 0.05
        super().__init__(params)

    @property
    def mu_s(self) -> float:
        """Return the static-friction coefficient.

        Returns:
            float: Current static-friction coefficient.

        """

        return self._mu_s

    @mu_s.setter
    def mu_s(self, value: float) -> None:
        """Set the static-friction coefficient.

        Args:
            value: New static-friction coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`mu_s` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`mu_s` must be non-negative.")
        self._mu_s = float(value)

    @property
    def mu_k(self) -> float:
        """Return the kinetic-friction coefficient.

        Returns:
            float: Current kinetic-friction coefficient.

        """

        return self._mu_k

    @mu_k.setter
    def mu_k(self, value: float) -> None:
        """Set the kinetic-friction coefficient.

        Args:
            value: New kinetic-friction coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is not positive.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`mu_k` must be a float or integer.")
        if value <= 0:
            raise e.ValueError("`mu_k` must be positive.")
        self._mu_k = float(value)

    @property
    def delta_t(self) -> float:
        """Return the time displacement.

        Returns:
            float: Current time displacement.

        """

        return self._delta_t

    @delta_t.setter
    def delta_t(self, value: float) -> None:
        """Set the time displacement.

        Args:
            value: New time displacement.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is negative.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`delta_t` must be a float or integer.")
        if value < 0:
            raise e.ValueError("`delta_t` must be non-negative.")
        self._delta_t = float(value)

    @property
    def alpha(self) -> float:
        """Return the speed coefficient.

        Returns:
            float: Current speed coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the speed coefficient.

        Args:
            value: New speed coefficient.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the canonical range.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`alpha` must be a float or integer.")
        if not 0.9 <= value <= 1:
            raise e.ValueError("`alpha` must be between 0.9 and 1.")
        self._alpha = float(value)

    @property
    def beta(self) -> float:
        """Return the random-displacement scale.

        Returns:
            float: Current random-displacement scale.

        """

        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the random-displacement scale.

        Args:
            value: New random-displacement scale.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is outside the canonical range.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`beta` must be a float or integer.")
        if not 0 < value <= 1:
            raise e.ValueError("`beta` must be greater than 0 and at most 1.")
        self._beta = float(value)

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one TWO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        n = pop.n_agents
        t = ctx.iteration + 1

        worst_fit = pop.fitness.max()
        best_fit = pop.fitness.min()
        weights = (pop.fitness - worst_fit) / (best_fit - worst_fit + c.EPSILON) + 1

        candidates = pop.positions.clone()
        current_mu_k = self.mu_k - (self.mu_k - 0.1) * (ctx.iteration / max(ctx.n_iterations, 1))
        for i in range(n):
            delta = torch.zeros_like(pop.positions[i])
            for j in range(n):
                if i == j or weights[i] >= weights[j]:
                    continue

                force = torch.maximum(weights[i] * self.mu_s, weights[j] * self.mu_s) - weights[i] * current_mu_k
                acceleration = force / _nonzero(weights[i] * pop.fitness.new_tensor(current_mu_k))
                acceleration = acceleration * (pop.positions[j] - pop.positions[i])
                noise = torch.randn_like(pop.positions[i])
                delta += 0.5 * acceleration * self.delta_t**2
                delta += self.alpha**t * self.beta * (pop.ub - pop.lb) * noise

            candidates[i] += delta

        for i in range(n):
            if torch.rand((), device=pop.device) < 0.5:
                candidates[i] = pop.best_position + torch.randn_like(candidates[i]) / t * (
                    pop.best_position - candidates[i]
                )

        candidates = candidates.clamp(min=pop.lb, max=pop.ub)
        candidate_fitness = fn(candidates)
        improved = candidate_fitness < pop.fitness
        pop.positions[improved] = candidates[improved]
        pop.fitness[improved] = candidate_fitness[improved]
        pop.update_best()


class WEO(Optimizer):
    """Water Evaporation Optimization."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the WEO optimizer.

        Args:
            params: Algorithm parameter overrides.

        """

        self.E_min = -3.5
        self.E_max = -0.5
        self.theta_min = -torch.pi / 3.6
        self.theta_max = -torch.pi / 9
        super().__init__(params)

    @property
    def E_min(self) -> float:
        """Return the minimum substrate energy.

        Returns:
            float: Current minimum substrate energy.

        """

        return self._E_min

    @E_min.setter
    def E_min(self, value: float) -> None:
        """Set the minimum substrate energy.

        Args:
            value: New minimum substrate energy.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`E_min` must be a float or integer.")
        self._E_min = float(value)

    @property
    def E_max(self) -> float:
        """Return the maximum substrate energy.

        Returns:
            float: Current maximum substrate energy.

        """

        return self._E_max

    @E_max.setter
    def E_max(self, value: float) -> None:
        """Set the maximum substrate energy.

        Args:
            value: New maximum substrate energy.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is below the minimum.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`E_max` must be a float or integer.")
        if value < self.E_min:
            raise e.ValueError("`E_max` must be greater than or equal to `E_min`.")
        self._E_max = float(value)

    @property
    def theta_min(self) -> float:
        """Return the minimum contact angle.

        Returns:
            float: Current minimum contact angle.

        """

        return self._theta_min

    @theta_min.setter
    def theta_min(self, value: float) -> None:
        """Set the minimum contact angle.

        Args:
            value: New minimum contact angle.

        Raises:
            TypeError: If the value is not numeric.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`theta_min` must be a float or integer.")
        self._theta_min = float(value)

    @property
    def theta_max(self) -> float:
        """Return the maximum contact angle.

        Returns:
            float: Current maximum contact angle.

        """

        return self._theta_max

    @theta_max.setter
    def theta_max(self, value: float) -> None:
        """Set the maximum contact angle.

        Args:
            value: New maximum contact angle.

        Raises:
            TypeError: If the value is not numeric.
            ValueError: If the value is below the minimum.

        """

        if not isinstance(value, (float, int)):
            raise e.TypeError("`theta_max` must be a float or integer.")
        if value < self.theta_min:
            raise e.ValueError("`theta_max` must be greater than or equal to `theta_min`.")
        self._theta_max = float(value)

    def compile(self, population) -> None:
        """Validate the population required by pairwise evaporation steps.

        Args:
            population: Population whose tensors define the optimizer state.

        Raises:
            SizeError: If fewer than two agents are available.

        """

        if population.n_agents < 2:
            raise e.SizeError("`population.n_agents` must be at least 2 for WEO.")

    def update(self, ctx: UpdateContext) -> None:
        """Advance the population by one WEO step.

        Args:
            ctx: Update context containing the population, objective, and iteration state.

        """

        pop = ctx.space.population
        fn = ctx.function
        device = pop.device
        n = pop.n_agents
        best_fit = pop.fitness.min()
        worst_fit = pop.fitness.max()
        denominator = (worst_fit - best_fit).clamp_min(torch.finfo(pop.dtype).eps)

        for i in range(n):
            normalized = (pop.fitness[i] - best_fit) / denominator
            if ctx.iteration <= max(ctx.n_iterations, 1) / 2:
                substrate_energy = (self.E_max - self.E_min) * normalized + self.E_min
                probability = torch.exp(substrate_energy)
            else:
                theta = (self.theta_max - self.theta_min) * normalized + self.theta_min
                cosine = torch.cos(theta)
                base = (2 / 3 + cosine**3 / 3 - cosine).clamp_min(torch.finfo(pop.dtype).eps)
                probability = (1 / 2.6) * base.pow(-2 / 3) * (1 - cosine)

            mask = (torch.rand_like(pop.positions[i]) < probability).to(pop.dtype)
            pair = torch.randperm(n, device=device)[:2]
            step = torch.rand((), device=device, dtype=pop.dtype) * (pop.positions[pair[0]] - pop.positions[pair[1]])
            candidate = (pop.positions[i] + step * mask).clamp(min=pop.lb, max=pop.ub)
            candidate_fit = fn(candidate.unsqueeze(0))[0]
            if candidate_fit < pop.fitness[i]:
                pop.positions[i] = candidate
                pop.fitness[i] = candidate_fit

        pop.update_best()
