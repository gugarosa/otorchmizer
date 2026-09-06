# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Standard multi-objective function wrapper."""

from __future__ import annotations

from collections.abc import Callable

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveFunction:
    """Wraps multiple objective functions.

    Notes:
        Returns a tensor shaped (n_agents, n_objectives) containing the fitness for each objective.

    """

    def __init__(self, functions: list[Callable], batch: bool = False) -> None:
        """Wrap each objective with the same batching strategy.

        Args:
            functions: List of objective callables.
            batch: If True, callables handle full population tensors.

        Raises:
            TypeError: Functions are not a list or an objective is not callable.

        """

        logger.info("Creating class: MultiObjectiveFunction.")

        if not isinstance(functions, list):
            raise e.TypeError("`functions` should be a list.")

        self.functions = [Function(f, batch=batch) for f in functions]
        self.built = True

        logger.debug("Functions: %d | Built: %s.", len(self.functions), self.built)
        logger.info("Class created.")

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates all objectives.

        Args:
            positions: Population tensor shaped (n_agents, n_variables, n_dimensions).

        Returns:
            Tensor of shape (n_agents, n_objectives).

        """

        return torch.stack([f(positions) for f in self.functions], dim=-1)
