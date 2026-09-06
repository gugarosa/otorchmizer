# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Boolean (binary) search space."""

from __future__ import annotations

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BooleanSpace(Space):
    """Search space for binary optimization."""

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        """Initialize a binary search space.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.

        Notes:
            Bounds are fixed to [0, 1], and positions are initialized with values from {0, 1}.

        """

        logger.info("Creating class: BooleanSpace.")

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=[0.0] * n_variables,
            upper_bound=[1.0] * n_variables,
            mapping=mapping,
            device=device,
        )

        self.build()

        logger.info("Class created.")

    def _initialize(self) -> None:
        self.population.initialize_binary()
