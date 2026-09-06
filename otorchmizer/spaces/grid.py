# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Grid-based search space."""

from __future__ import annotations

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


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
    ) -> None:
        """Initialize a grid search space.

        Args:
            n_variables: Number of decision variables.
            step: Positive step size for each decision variable.
            lower_bound: Lower bound for each decision variable.
            upper_bound: Upper bound for each decision variable.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.

        Raises:
            SizeError: If `step` does not contain one value per decision variable.
            ValueError: If `step` contains a nonfinite or nonpositive value.

        Notes:
            The Cartesian product of the bounded ranges determines the number of agents.

        """

        logger.info("Creating class: GridSpace.")

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
        if self.step.shape != (n_variables,):
            raise e.SizeError(f"`step` must have shape {(n_variables,)}, but got {tuple(self.step.shape)}.")
        if not torch.isfinite(self.step).all() or (self.step <= 0).any():
            raise e.ValueError("`step` must contain finite positive values.")

        self._create_grid()
        self.build()

        logger.info("Class created.")

    def _create_grid(self) -> None:
        lb = self.population.lb.squeeze(-1)
        ub = self.population.ub.squeeze(-1)
        step = self.step

        ranges = [
            torch.arange(
                lb[i].item(),
                ub[i].item() + step[i].item(),
                step[i].item(),
                device=self.device,
            )
            for i in range(self.population.n_variables)
        ]
        ranges = [values[values <= ub[i]] for i, values in enumerate(ranges)]

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
        )

        self.grid = grid

    def _initialize(self) -> None:
        self.population.initialize_static(self.grid)
