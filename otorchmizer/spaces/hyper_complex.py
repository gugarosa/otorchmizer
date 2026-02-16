"""Hypercomplex search space."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class HyperComplexSpace(Space):
    """Search space for hypercomplex (multi-dimensional) optimization.

    Each variable has n_dimensions > 1 (e.g., quaternions use 4).
    Bounds are fixed to [0, 1] in the hypercomplex domain;
    mapping to real bounds is done via the hyper math module.
    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
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
