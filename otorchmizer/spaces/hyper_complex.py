# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Hypercomplex search space."""

from __future__ import annotations

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class HyperComplexSpace(Space):
    """Search space for hypercomplex optimization."""

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        """Initialize a hypercomplex search space.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions in each hypercomplex variable.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.

        Notes:
            Bounds are fixed to [0, 1] in the hypercomplex domain. The hyper math module maps values to real bounds.

        """

        logger.info("Creating class: HyperComplexSpace.")

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            lower_bound=[0.0] * n_variables,
            upper_bound=[1.0] * n_variables,
            mapping=mapping,
            device=device,
        )

        self.build()

        logger.info("Class created.")

    def _initialize(self) -> None:
        self.population.initialize_uniform()
