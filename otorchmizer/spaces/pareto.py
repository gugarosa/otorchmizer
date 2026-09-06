# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Pareto-frontier search space."""

from __future__ import annotations

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ParetoSpace(Space):
    """Search space for multi-objective optimization with preloaded data points."""

    def __init__(
        self,
        data_points: torch.Tensor,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        """Initialize a Pareto search space.

        Args:
            data_points: Predefined data with shape (n_agents, n_variables).
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.

        Notes:
            Agents are initialized from `data_points` instead of random samples, and bound clipping is disabled.
            Floating-point data retains its input dtype.

        """

        logger.info("Creating class: ParetoSpace.")

        n_agents, n_variables = data_points.shape

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=[0.0] * n_variables,
            upper_bound=[0.0] * n_variables,
            mapping=mapping,
            device=device,
        )

        if data_points.is_floating_point():
            self.population.to(self.device, dtype=data_points.dtype)
        self._data_points = data_points.to(self.device)
        self.build()

        logger.info("Class created.")

    def _initialize(self) -> None:
        self.population.initialize_static(self._data_points)

    def clip(self) -> None:
        """Leave Pareto-space positions unchanged."""

        pass
