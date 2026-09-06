# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Population — batched tensor storage for all agents.

Replaces the per-object Agent + List[Agent] pattern from Opytimizer
with a single contiguous tensor block for GPU-friendly access.

"""

from __future__ import annotations

from math import inf

import torch

import otorchmizer.utils.exception as e
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


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
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        mapping: list[str] | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Allocate population storage, bounds, and best-candidate tracking.

        Args:
            n_agents: Number of candidate solutions.
            n_variables: Number of decision variables per agent.
            n_dimensions: Dimensionality per variable (1 for standard, >1 for hypercomplex).
            lower_bound: Lower bounds shaped (n_variables,) or (n_variables, n_dimensions).
            upper_bound: Upper bounds shaped (n_variables,) or (n_variables, n_dimensions).
            mapping: Human-readable variable names.
            device: Target device for all tensors.
            dtype: Floating-point dtype for positions, bounds, and fitness.

        Raises:
            ValueError: An agent, variable, or dimension count is not positive.

        Notes:
            Float16 and bfloat16 storage are supported for mixed-precision workflows.
            One-dimensional bounds are expanded for broadcasting across each variable's dimensions.

        """

        if n_agents <= 0:
            raise e.ValueError("`n_agents` should be > 0.")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0.")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0.")

        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.device = device
        self.dtype = dtype

        self.positions = torch.zeros(n_agents, n_variables, n_dimensions, device=device, dtype=dtype)
        self.fitness = torch.full((n_agents,), inf, device=device, dtype=dtype)

        self.lb = lower_bound.to(device=device, dtype=dtype)
        self.ub = upper_bound.to(device=device, dtype=dtype)

        if self.lb.dim() == 1:
            self.lb = self.lb.unsqueeze(-1)
        if self.ub.dim() == 1:
            self.ub = self.ub.unsqueeze(-1)

        self.best_position = torch.zeros(n_variables, n_dimensions, device=device, dtype=dtype)
        self.best_fitness = torch.tensor(inf, device=device, dtype=dtype)

        self.mapping = mapping or [f"x{i}" for i in range(n_variables)]

    def clip(self) -> None:
        """Clamps all positions to bound limits — fully vectorized."""

        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        self.positions = self.positions.clamp(min=lb, max=ub)

    def initialize_uniform(self) -> None:
        """Fills all positions uniformly within bounds — single tensor op."""

        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        self.positions = torch.rand_like(self.positions) * (ub - lb) + lb
        self.best_position = self.positions[0].clone()

    def initialize_binary(self) -> None:
        """Fills all positions with binary {0, 1} values."""

        self.positions = torch.round(torch.rand_like(self.positions))
        self.best_position = self.positions[0].clone()

    def initialize_static(self, values: torch.Tensor) -> None:
        """Fills positions from pre-defined values.

        Args:
            values: Positions shaped (n_agents, n_variables) or (n_agents, n_variables, n_dimensions).

        Raises:
            SizeError: Values do not match the population shape after adding a singleton dimension if needed.

        Notes:
            Two-dimensional values are accepted only when the population has one dimension per variable.
            Values are moved to the population's device and dtype, and the first row seeds the best position.

        """

        if values.dim() == 2:
            values = values.unsqueeze(-1)
        expected = (self.n_agents, self.n_variables, self.n_dimensions)
        if tuple(values.shape) != expected:
            raise e.SizeError(f"`values` must have shape {expected}, but got {tuple(values.shape)}.")

        self.positions = values.to(device=self.device, dtype=self.dtype)
        self.best_position = self.positions[0].clone()

    def update_best(self) -> None:
        """Finds population-wide best — vectorized argmin."""

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

        dt = dtype or self.dtype
        self.device = device
        self.dtype = dt
        self.positions = self.positions.to(device=device, dtype=dt)
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
            Bounds and current best-tracking values are retained on each target device.

        """

        n = len(devices)
        if not 1 <= n <= self.n_agents:
            raise e.ValueError(f"`devices` must contain between 1 and {self.n_agents} targets, but got {n}.")

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
            pop.positions = ch_pos.to(dev)
            pop.fitness = ch_fit.to(dev)
            pop.best_position = self.best_position.to(dev)
            pop.best_fitness = self.best_fitness.to(dev)
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
            ValueError: No populations are supplied.

        """

        if not populations:
            raise e.ValueError("`populations` must contain at least one population.")

        ref = populations[0]
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
        best_idx = best_fitnesses.argmin().item()
        merged.best_fitness = populations[best_idx].best_fitness.to(target_device)
        merged.best_position = populations[best_idx].best_position.to(target_device)
        return merged
