# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Population — batched tensor storage for all agents.

Replaces the per-object Agent + List[Agent] pattern from Opytimizer
with a single contiguous tensor block for GPU-friendly access.

"""

from __future__ import annotations

from math import inf
from operator import index

import torch


class Population:
    """Stores the entire population as contiguous tensors.

    Notes:
        Positions live in batched tensors shaped (n_agents, n_variables, n_dimensions),
        enabling vectorized operations and GPU parallelism.
        Unscored fitness and best fitness start at positive infinity in the requested dtype.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        lower_bound: float | list | tuple | torch.Tensor,
        upper_bound: float | list | tuple | torch.Tensor,
        mapping: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate population storage, bounds, and best-candidate tracking.

        Args:
            n_agents: Number of candidate solutions.
            n_variables: Number of decision variables per agent.
            n_dimensions: Dimensionality per variable (1 for standard, >1 for hypercomplex).
            lower_bound: Scalar or bounds shaped (n_variables,), (n_variables, 1), or (n_variables, n_dimensions).
            upper_bound: Scalar or bounds shaped (n_variables,), (n_variables, 1), or (n_variables, n_dimensions).
            mapping: One unique string name per variable.
            device: Target device for all tensors.
            dtype: Floating-point storage dtype, or None to use the PyTorch default.

        Raises:
            TypeError: A count, mapping, or dtype has an invalid type.
            ValueError: Bound shapes do not match the configured dimensions.
            ValueError: Counts, bounds, mapping, or dtype have invalid values.

        Notes:
            Float16 and bfloat16 storage are supported for mixed-precision workflows.
            One-dimensional bounds are expanded for broadcasting across each variable's dimensions.

        """

        counts = []
        for name, value in (("n_agents", n_agents), ("n_variables", n_variables), ("n_dimensions", n_dimensions)):
            try:
                count = index(value)
            except TypeError as error:
                raise TypeError(f"`{name}` must be an integer.") from error
            if isinstance(value, bool):
                raise TypeError(f"`{name}` must be an integer, not a boolean.")
            if count <= 0:
                raise ValueError(f"`{name}` must be positive.")
            counts.append(count)
        n_agents, n_variables, n_dimensions = counts
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not isinstance(dtype, torch.dtype):
            raise TypeError("`dtype` must be a torch.dtype.")
        if not dtype.is_floating_point:
            raise ValueError("`dtype` must be floating point.")
        device = torch.device(device)

        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.device = device
        self.dtype = dtype

        bounds = []
        for name, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
            bound = torch.as_tensor(value, device=device, dtype=dtype)
            if bound.ndim == 0:
                bound = bound.expand(n_variables)
            if bound.shape == (n_variables,):
                bound = bound.unsqueeze(-1)
            if bound.shape not in ((n_variables, 1), (n_variables, n_dimensions)):
                raise ValueError(f"`{name}` must match the variable and dimension counts.")
            if not torch.isfinite(bound).all():
                raise ValueError(f"`{name}` must contain finite values.")
            bounds.append(bound.clone())
        self.lb, self.ub = bounds
        if (self.lb > self.ub).any():
            raise ValueError("`lower_bound` must not exceed `upper_bound`.")

        if mapping is None:
            mapping = [f"x{i}" for i in range(n_variables)]
        if not isinstance(mapping, list) or any(not isinstance(name, str) for name in mapping):
            raise TypeError("`mapping` must be a list of strings.")
        if len(mapping) != n_variables or len(set(mapping)) != n_variables:
            raise ValueError("`mapping` must contain one unique name per variable.")
        self.mapping = mapping.copy()
        self.positions = torch.zeros(n_agents, n_variables, n_dimensions, device=device, dtype=dtype)
        self.device = self.positions.device
        self._reset_fitness()

    def _reset_fitness(self) -> None:
        self.fitness = self.positions.new_full((self.n_agents,), inf)
        self.best_fitness = self.positions.new_tensor(inf)
        self.best_position = self.positions[0].clone()

    def clip(self) -> None:
        """Clamps all positions to bound limits — fully vectorized."""

        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        self.positions = self.positions.clamp(min=lb, max=ub)

    def initialize_uniform(self) -> None:
        """Sample positions uniformly within bounds and reset all scores and the best archive."""

        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        self.positions = torch.rand_like(self.positions) * (ub - lb) + lb
        self._reset_fitness()

    def initialize_binary(self) -> None:
        """Sample binary positions and reset all scores and the best archive."""

        self.positions = torch.round(torch.rand_like(self.positions))
        self._reset_fitness()

    def initialize_static(self, values: torch.Tensor) -> None:
        """Fills positions from pre-defined values.

        Args:
            values: Positions shaped (n_agents, n_variables) or (n_agents, n_variables, n_dimensions).

        Raises:
            ValueError: Values do not match the population shape after adding a singleton dimension if needed.

        Notes:
            Two-dimensional values are accepted only when the population has one dimension per variable.
            Values are copied to the population's device and dtype, and the first row seeds the best position.
            Current and best fitness reset to positive infinity, starting a fresh population.

        """

        if values.dim() == 2:
            values = values.unsqueeze(-1)
        expected = (self.n_agents, self.n_variables, self.n_dimensions)
        if tuple(values.shape) != expected:
            raise ValueError(f"`values` must have shape {expected}, but got {tuple(values.shape)}.")

        self.positions = values.to(device=self.device, dtype=self.dtype).clone()
        self._reset_fitness()

    def update_best(self) -> None:
        """Archive strict improvements without accepting NaN fitness.

        Raises:
            ValueError: Fitness does not contain one scalar per agent.
            ValueError: Fitness contains NaN.

        """

        if self.fitness.shape != (self.n_agents,):
            raise ValueError("`fitness` must contain one scalar per agent.")
        if torch.isnan(self.fitness).any():
            raise ValueError("`fitness` must not contain NaN.")
        best_idx = self.fitness.argmin()
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx].clone()
            self.best_position = self.positions[best_idx].clone()

    def clone_positions(self) -> torch.Tensor:
        """Clone all positions without a Python deep copy.

        Returns:
            Independent position tensor with the population's shape, device, and dtype.

        """

        return self.positions.clone()

    def clone_fitness(self) -> torch.Tensor:
        """Clone all fitness values.

        Returns:
            Independent fitness tensor with one value per agent.

        """

        return self.fitness.clone()

    def sort_by_fitness(self) -> torch.Tensor:
        """Reorder positions and fitness by ascending fitness.

        Returns:
            Original agent indices in sorted order.

        """

        sorted_idx = torch.argsort(self.fitness)
        self.positions = self.positions[sorted_idx]
        self.fitness = self.fitness[sorted_idx]
        return sorted_idx

    @property
    def mapped_positions(self) -> dict[str, torch.Tensor]:
        """Dictionary mapping variable names to their position tensors."""

        return {m: self.positions[:, i, :] for i, m in enumerate(self.mapping)}

    def __repr__(self) -> str:
        return (
            f"Population(n_agents={self.n_agents}, n_variables={self.n_variables}, "
            f"n_dimensions={self.n_dimensions}, device={self.device}, dtype={self.dtype})"
        )

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> Population:
        """Moves all tensors to *device* (and optionally casts to *dtype*).

        Args:
            device: Target device.
            dtype: Optional target dtype.

        Returns:
            self (for chaining).

        """

        dt = self.dtype if dtype is None else dtype
        if not isinstance(dt, torch.dtype):
            raise TypeError("`dtype` must be a torch.dtype.")
        if not dt.is_floating_point:
            raise ValueError("`dtype` must be floating point.")
        device = torch.device(device)
        self.positions = self.positions.to(device=device, dtype=dt)
        self.device = self.positions.device
        self.dtype = dt
        self.fitness = self.fitness.to(device=device, dtype=dt)
        self.best_position = self.best_position.to(device=device, dtype=dt)
        self.best_fitness = self.best_fitness.to(device=device, dtype=dt)
        self.lb = self.lb.to(device=device, dtype=dt)
        self.ub = self.ub.to(device=device, dtype=dt)
        return self

    def scatter(self, devices: list[torch.device]) -> list[Population]:
        """Splits this population across multiple devices (multi-GPU).

        Args:
            devices: List of target devices.

        Returns:
            List of Population instances, one per device.

        Raises:
            ValueError: The number of devices is zero or exceeds the number of agents.

        Notes:
            Each sub-population receives an equal or nearly equal share of agents.
            Bounds, positions, fitness, and best-tracking tensors are independently owned on each target device.

        """

        n = len(devices)
        if not 1 <= n <= self.n_agents:
            raise ValueError(f"`devices` must contain between 1 and {self.n_agents} targets, but got {n}.")

        pos_chunks = self.positions.tensor_split(n, dim=0)
        fit_chunks = self.fitness.tensor_split(n, dim=0)
        pops = []
        for ch_pos, ch_fit, dev in zip(pos_chunks, fit_chunks, devices):
            pop = Population(
                n_agents=ch_pos.shape[0],
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                lower_bound=self.lb.squeeze(-1),
                upper_bound=self.ub.squeeze(-1),
                mapping=self.mapping,
                device=dev,
                dtype=self.dtype,
            )
            pop.positions = ch_pos.to(dev).clone()
            pop.fitness = ch_fit.to(dev).clone()
            pop.best_position = self.best_position.to(dev).clone()
            pop.best_fitness = self.best_fitness.to(dev).clone()
            pops.append(pop)
        return pops

    @staticmethod
    def gather(populations: list[Population], target_device: torch.device) -> Population:
        """Merges sub-populations from multiple devices back into one.

        Args:
            populations: Sub-populations (potentially on different devices).
            target_device: Device for the merged population.

        Returns:
            Merged Population on *target_device*.

        Raises:
            ValueError: Populations are empty or have incompatible metadata or NaN archived fitness.

        """

        if not populations:
            raise ValueError("`populations` must contain at least one population.")

        ref = populations[0]
        for population in populations[1:]:
            if (
                (population.n_variables, population.n_dimensions, population.dtype)
                != (ref.n_variables, ref.n_dimensions, ref.dtype)
                or population.mapping != ref.mapping
                or not torch.equal(population.lb.to(ref.device), ref.lb)
                or not torch.equal(population.ub.to(ref.device), ref.ub)
            ):
                raise ValueError("`populations` must have matching shapes, dtype, bounds, and mapping.")
        total_agents = sum(p.n_agents for p in populations)
        merged = Population(
            n_agents=total_agents,
            n_variables=ref.n_variables,
            n_dimensions=ref.n_dimensions,
            lower_bound=ref.lb.squeeze(-1).to(target_device),
            upper_bound=ref.ub.squeeze(-1).to(target_device),
            mapping=ref.mapping,
            device=target_device,
            dtype=ref.dtype,
        )
        merged.positions = torch.cat([p.positions.to(target_device) for p in populations], dim=0)
        merged.fitness = torch.cat([p.fitness.to(target_device) for p in populations], dim=0)

        best_fitnesses = torch.stack([p.best_fitness.to(target_device) for p in populations])
        if torch.isnan(best_fitnesses).any():
            raise ValueError("`best_fitness` must not contain NaN.")
        best_idx = best_fitnesses.argmin().item()
        merged.best_fitness = populations[best_idx].best_fitness.to(target_device).clone()
        merged.best_position = populations[best_idx].best_position.to(target_device).clone()
        return merged
