"""Population — batched tensor storage for all agents.

Replaces the per-object Agent + List[Agent] pattern from Opytimizer
with a single contiguous tensor block for GPU-friendly access.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class Population:
    """Stores the entire population as contiguous tensors.

    All agent data lives in batched tensors of shape
    (n_agents, n_variables, n_dimensions), enabling vectorized
    operations and GPU parallelism.
    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        mapping: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of candidate solutions.
            n_variables: Number of decision variables per agent.
            n_dimensions: Dimensionality per variable (1 for standard, >1 for hypercomplex).
            lower_bound: Per-variable lower bounds, shape (n_variables,).
            upper_bound: Per-variable upper bounds, shape (n_variables,).
            mapping: Human-readable variable names.
            device: Target device for all tensors.
        """

        if n_agents <= 0:
            raise e.ValueError("`n_agents` should be > 0")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0")

        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.device = device

        self.positions = torch.zeros(
            n_agents, n_variables, n_dimensions, device=device
        )
        self.fitness = torch.full(
            (n_agents,), c.FLOAT_MAX, device=device
        )

        self.lb = lower_bound.to(device=device, dtype=torch.float32)
        self.ub = upper_bound.to(device=device, dtype=torch.float32)

        if self.lb.dim() == 1:
            self.lb = self.lb.unsqueeze(-1)  # (n_variables, 1)
        if self.ub.dim() == 1:
            self.ub = self.ub.unsqueeze(-1)  # (n_variables, 1)

        self.best_position = torch.zeros(
            n_variables, n_dimensions, device=device
        )
        self.best_fitness = torch.tensor(c.FLOAT_MAX, device=device)

        self.mapping = mapping or [f"x{i}" for i in range(n_variables)]

    def clip(self) -> None:
        """Clamps all positions to bound limits — fully vectorized."""

        lb = self.lb.unsqueeze(0)  # (1, n_variables, 1) or (1, n_variables, n_dims)
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
            values: Tensor of shape (n_agents, n_variables) or
                    (n_agents, n_variables, n_dimensions).
        """

        if values.dim() == 2:
            values = values.unsqueeze(-1)
        self.positions = values.to(device=self.device, dtype=torch.float32)
        self.best_position = self.positions[0].clone()

    def update_best(self) -> None:
        """Finds population-wide best — vectorized argmin."""

        best_idx = self.fitness.argmin()
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx].clone()
            self.best_position = self.positions[best_idx].clone()

    def clone_positions(self) -> torch.Tensor:
        """Returns a clone of all positions (replaces deepcopy)."""

        return self.positions.clone()

    def clone_fitness(self) -> torch.Tensor:
        """Returns a clone of all fitness values."""

        return self.fitness.clone()

    def sort_by_fitness(self) -> torch.Tensor:
        """Sorts population by fitness (ascending). Returns sorted indices."""

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
            f"n_dimensions={self.n_dimensions}, device={self.device})"
        )
