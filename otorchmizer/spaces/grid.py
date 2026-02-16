"""Grid-based search space."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class GridSpace(Space):
    """Search space that exhaustively evaluates a grid of values.

    Creates all combinations of values defined by step sizes and bounds.
    The number of agents is determined by the grid size.
    """

    def __init__(
        self,
        n_variables: int,
        step: Union[float, List, Tuple, torch.Tensor],
        lower_bound: Union[float, List, Tuple, torch.Tensor],
        upper_bound: Union[float, List, Tuple, torch.Tensor],
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        logger.info("Creating class: GridSpace.")

        # Initialize with n_agents=1 (will be overridden by grid creation)
        super().__init__(
            n_agents=1,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=device,
        )

        step_t = self._to_tensor(step, n_variables)
        self.step = step_t.to(self.device)

        self._create_grid()
        self.build()

        logger.info("Class created.")

    def _create_grid(self) -> None:
        """Creates a grid of all possible search values using torch.meshgrid."""

        lb = self.population.lb.squeeze(-1)
        ub = self.population.ub.squeeze(-1)
        step = self.step

        ranges = [
            torch.arange(lb[i].item(), ub[i].item() + step[i].item(), step[i].item(),
                         device=self.device)
            for i in range(self.population.n_variables)
        ]

        mesh = torch.meshgrid(*ranges, indexing="ij")
        grid = torch.stack([m.ravel() for m in mesh], dim=1)  # (n_grid, n_vars)

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
        )

        self.grid = grid

    def _initialize(self) -> None:
        """Initializes agents at grid positions."""

        self.population.initialize_static(self.grid)
