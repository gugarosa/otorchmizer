"""Function wrappers for objective functions with optional auto-batching."""

from __future__ import annotations

import functools
from typing import Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class Function:
    """Wraps a user objective function with optional batch support.

    When batch=False (default), the function is expected to take a single
    agent's position (n_variables, n_dimensions) and return a scalar.
    It is automatically vectorized via torch.vmap for parallel evaluation.

    When batch=True, the function must accept the full population tensor
    (n_agents, n_variables, n_dimensions) and return (n_agents,).
    """

    def __init__(self, pointer: callable, batch: bool = False) -> None:
        """Initialization method.

        Args:
            pointer: Callable returning a fitness value.
            batch: If True, pointer handles full population tensors natively.
        """

        logger.info("Creating class: Function.")

        if not callable(pointer):
            raise e.TypeError("`pointer` should be a callable")

        self._raw_pointer = pointer
        self.batch = batch

        if hasattr(pointer, "__name__"):
            self.name = pointer.__name__
        else:
            self.name = pointer.__class__.__name__

        if batch:
            self._fn = pointer
        else:
            # torch.vmap vectorizes a single-agent function across the batch dim
            try:
                self._fn = torch.vmap(pointer)
            except Exception:
                # Fallback: manual loop for functions that don't support vmap
                self._fn = self._manual_vmap(pointer)

        self.built = True

        logger.debug("Function: %s | Batch: %s | Built: %s.", self.name, batch, self.built)
        logger.info("Class created.")

    @staticmethod
    def _manual_vmap(fn: callable) -> callable:
        """Manual fallback for functions incompatible with torch.vmap.

        Args:
            fn: Single-agent function.

        Returns:
            Batched version that loops over the first dimension.
        """

        @functools.wraps(fn)
        def batched(positions: torch.Tensor) -> torch.Tensor:
            results = torch.stack([fn(positions[i]) for i in range(positions.shape[0])])
            return results

        return batched

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates positions and returns fitness values.

        Args:
            positions: Tensor of shape (n_agents, n_variables, n_dimensions).

        Returns:
            Fitness tensor of shape (n_agents,).
        """

        return self._fn(positions)

    def __repr__(self) -> str:
        return f"Function(name={self.name}, batch={self.batch})"
