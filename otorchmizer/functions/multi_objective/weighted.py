"""Weighted multi-objective function wrapper."""

from __future__ import annotations

from typing import List

import torch

import otorchmizer.utils.exception as e
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """Scalarizes multiple objectives via weighted sum.

    Returns a single fitness value: z = Σ(wᵢ · fᵢ(x)).
    """

    def __init__(self, functions: List[callable], weights: List[float],
                 batch: bool = False) -> None:
        """Initialization method.

        Args:
            functions: List of objective callables.
            weights: Per-objective weights for scalarization.
            batch: If True, callables handle full population tensors.
        """

        logger.info("Creating class: MultiObjectiveWeightedFunction.")

        super().__init__(functions, batch)

        if not isinstance(weights, list):
            raise e.TypeError("`weights` should be a list")
        if len(weights) != len(self.functions):
            raise e.SizeError("`weights` should have the same size as `functions`")

        self.weights = torch.tensor(weights, dtype=torch.float32)

        logger.debug("Weights: %s.", weights)
        logger.info("Class created.")

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates weighted sum of objectives.

        Args:
            positions: (n_agents, n_variables, n_dimensions).

        Returns:
            Scalarized fitness tensor of shape (n_agents,).
        """

        objectives = super().__call__(positions)  # (n_agents, n_objectives)
        w = self.weights.to(objectives.device)
        return (objectives * w).sum(dim=-1)
