"""Standard continuous search space."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class SearchSpace(Space):
    """Standard search space for continuous optimization.

    Agents are initialized uniformly within bounds.
    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: Union[float, List, Tuple, torch.Tensor],
        upper_bound: Union[float, List, Tuple, torch.Tensor],
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        logger.info("Creating class: SearchSpace.")

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=device,
        )

        self.build()

        logger.info("Class created.")

    def _initialize(self) -> None:
        self.population.initialize_uniform()
