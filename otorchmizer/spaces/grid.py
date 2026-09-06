# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Grid-based search space."""

from __future__ import annotations

import math

import torch

from otorchmizer.core.space import Space


class GridSpace(Space):
    """Search space that exhaustively evaluates a bounded grid."""

    def __init__(
        self,
        n_variables: int,
        step: float | list[float] | tuple[float, ...] | torch.Tensor,
        lower_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        upper_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a grid search space.

        Args:
            n_variables: Number of decision variables.
            step: Positive step size for each decision variable.
            lower_bound: Lower bound for each decision variable.
            upper_bound: Upper bound for each decision variable.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.
            dtype: Storage dtype, or None to use the PyTorch default.

        Raises:
            ValueError: If `step` does not contain one value per decision variable.
            ValueError: If `step` contains a nonfinite or nonpositive value.

        Notes:
            The Cartesian product of the bounded ranges determines the number of agents.
            Coordinates use wider intermediate arithmetic before conversion to the population's storage dtype.

        """

        super().__init__(
            n_agents=1,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=device,
            dtype=dtype,
        )

        self.step = torch.as_tensor(step, device=self.device, dtype=self.population.dtype)
        if self.step.ndim == 0:
            self.step = self.step.expand(n_variables)
        if self.step.shape != (n_variables,):
            raise ValueError(f"`step` must have shape {(n_variables,)}, but got {tuple(self.step.shape)}.")
        if not torch.isfinite(self.step).all() or (self.step <= 0).any():
            raise ValueError("`step` must contain finite positive values.")

        self._create_grid()
        self.build()

    def _create_grid(self) -> None:
        lb = self.population.lb.squeeze(-1)
        ub = self.population.ub.squeeze(-1)
        step = self.step
        working_dtype = torch.float32 if self.population.dtype == torch.float16 else torch.float64

        ranges = []
        for lower, upper, spacing in zip(lb, ub, step):
            low, high, stride = lower.item(), upper.item(), spacing.item()
            n_steps = math.ceil((high - low) / stride)
            values = low + stride * torch.arange(n_steps + 1, device=self.device, dtype=working_dtype)
            tolerance = min(stride / 2, 2 * torch.finfo(self.population.dtype).eps * max(abs(low), abs(high)))
            values = values[(values <= upper) | ((values - upper).abs() <= tolerance)]
            values = torch.where((values - high).abs() <= tolerance, high, values)
            ranges.append(values.to(dtype=self.population.dtype))

        mesh = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack([m.ravel() for m in mesh], dim=1)

        n_grid = grid.shape[0]
        new_lb = self.population.lb
        new_ub = self.population.ub

        self.population = type(self.population)(
            n_agents=n_grid,
            n_variables=self.population.n_variables,
            n_dimensions=1,
            lower_bound=new_lb.squeeze(-1),
            upper_bound=new_ub.squeeze(-1),
            mapping=self.population.mapping,
            device=self.device,
            dtype=self.population.dtype,
        )

        self.grid = grid

    def _initialize(self) -> None:
        self.population.initialize_static(self.grid)
