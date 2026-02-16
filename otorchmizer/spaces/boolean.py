"""Boolean (binary) search space."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class BooleanSpace(Space):
    """Search space for binary optimization.

    Bounds are fixed to [0, 1] and positions are initialized as {0, 1}.
    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
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
