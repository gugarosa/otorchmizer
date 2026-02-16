"""Pareto-frontier search space."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ParetoSpace(Space):
    """Search space for multi-objective optimization with pre-loaded data points.

    Agents are initialized from given data points rather than random sampling.
    No bound clipping is performed.
    """

    def __init__(
        self,
        data_points: torch.Tensor,
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        """Initialization method.

        Args:
            data_points: Pre-defined data of shape (n_agents, n_variables).
            mapping: Human-readable variable names.
            device: Target device.
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

        self._data_points = data_points.to(self.device)
        self.build()

        logger.info("Class created.")

    def _initialize(self) -> None:
        self.population.initialize_static(self._data_points)

    def clip(self) -> None:
        """No clipping for Pareto spaces."""

        pass
