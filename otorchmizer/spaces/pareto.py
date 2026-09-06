# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Pareto-frontier search space."""

from __future__ import annotations

import torch

from otorchmizer.core.space import Space


class ParetoSpace(Space):
    """Search space for multi-objective optimization with preloaded data points."""

    def __init__(
        self,
        data_points: torch.Tensor,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a Pareto search space.

        Args:
            data_points: Predefined data with shape (n_agents, n_variables).
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.
            dtype: Storage dtype, or None to preserve floating-point input precision.

        Notes:
            Agents are initialized from `data_points` instead of random samples, and bound clipping is disabled.
            Floating-point data retains its input dtype.

        """

        if not isinstance(data_points, torch.Tensor):
            raise TypeError("`data_points` must be a tensor.")
        if data_points.ndim != 2:
            raise ValueError("`data_points` must have shape (n_agents, n_variables).")
        n_agents, n_variables = data_points.shape
        if dtype is None and data_points.is_floating_point():
            dtype = data_points.dtype

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=[0.0] * n_variables,
            upper_bound=[0.0] * n_variables,
            mapping=mapping,
            device=device,
            dtype=dtype,
        )

        self._data_points = data_points.to(device=self.device, dtype=self.population.dtype).clone()
        self.build()

    def _initialize(self) -> None:
        self.population.initialize_static(self._data_points)

    def clip(self) -> None:
        """Leave Pareto-space positions unchanged."""

        pass
