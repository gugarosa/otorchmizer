# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Weighted multi-objective function wrapper."""

from __future__ import annotations

from collections.abc import Callable

import torch

import otorchmizer.utils.exception as e
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """Scalarizes multiple objectives via weighted sum.

    Notes:
        Returns one fitness value per agent using z = Σ(wᵢ · fᵢ(x)).
        Weights are stored in float32 and moved to the objective tensor's device during evaluation.

    """

    def __init__(self, functions: list[Callable], weights: list[float], batch: bool = False) -> None:
        """Wrap objectives and retain one scalarization weight per objective.

        Args:
            functions: List of objective callables.
            weights: Per-objective weights for scalarization.
            batch: If True, callables handle full population tensors.

        Raises:
            TypeError: Functions or weights are not lists, or an objective is not callable.
            SizeError: The number of weights differs from the number of objectives.

        """

        logger.info("Creating class: MultiObjectiveWeightedFunction.")

        super().__init__(functions, batch)

        if not isinstance(weights, list):
            raise e.TypeError("`weights` should be a list.")
        if len(weights) != len(self.functions):
            raise e.SizeError("`weights` should have the same size as `functions`.")

        self.weights = torch.tensor(weights, dtype=torch.float32)

        logger.debug("Weights: %s.", weights)
        logger.info("Class created.")

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates weighted sum of objectives.

        Args:
            positions: Population tensor shaped (n_agents, n_variables, n_dimensions).

        Returns:
            Scalarized fitness tensor of shape (n_agents,).

        """

        objectives = super().__call__(positions)
        w = self.weights.to(objectives.device)
        return (objectives * w).sum(dim=-1)
