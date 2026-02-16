"""Standard multi-objective function wrapper."""

from __future__ import annotations

from typing import List

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveFunction:
    """Wraps multiple objective functions.

    Returns a tensor of shape (n_agents, n_objectives) containing
    the fitness for each objective.
    """

    def __init__(self, functions: List[callable], batch: bool = False) -> None:
        """Initialization method.

        Args:
            functions: List of objective callables.
            batch: If True, callables handle full population tensors.
        """

        logger.info("Creating class: MultiObjectiveFunction.")

        if not isinstance(functions, list):
            raise e.TypeError("`functions` should be a list")

        self.functions = [Function(f, batch=batch) for f in functions]
        self.built = True

        logger.debug("Functions: %d | Built: %s.", len(self.functions), self.built)
        logger.info("Class created.")

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates all objectives.

        Args:
            positions: (n_agents, n_variables, n_dimensions).

        Returns:
            Tensor of shape (n_agents, n_objectives).
        """

        return torch.stack([f(positions) for f in self.functions], dim=-1)
