# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Objective functions with explicit batch contracts and scalar evaluation fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch._dynamo import is_compiling

from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


def _reject_nan(fitness: torch.Tensor) -> None:
    valid = ~torch.isnan(fitness).any()
    if is_compiling():
        # A host assertion preserves CUDA context usability after invalid objective values
        torch._assert_async(valid.cpu(), "`fitness` must not contain NaN.")
    elif not valid:
        raise ValueError("`fitness` must not contain NaN.")


class Function:
    """Wrap a user objective function for population evaluation."""

    def __init__(self, pointer: Callable, batch: bool = False) -> None:
        """Initialize an objective and its batching strategy.

        Args:
            pointer: Callable returning scalar fitness or a native batch of fitness values.
            batch: Whether the callable accepts an entire population rather than one agent.

        Raises:
            TypeError: The objective is not callable.

        Notes:
            Scalar objectives receive ``(n_variables, n_dimensions)`` tensors and are vectorized with ``torch.vmap``.
            Known vmap incompatibilities trigger a warning and a cached per-agent fallback.
            Other objective errors propagate without retrying. Objectives should be free of side effects,
            since a rejected vectorized call may execute part of the objective before fallback.
            Native batch callables receive ``(n_agents, n_variables, n_dimensions)`` and must return ``(n_agents,)``.
            Checkpoints retain the original callable and rebuild transient vectorization wrappers when loaded.
            Assigning batch rebuilds the wrapper and resets cached manual fallback.
            Compiled NaN checks raise RuntimeError using a host assertion, without invalidating the CUDA context.
            This host validation synchronizes CUDA evaluation and is not CUDA-Graph-capture compatible.

        """

        if not callable(pointer):
            raise TypeError("`pointer` must be callable.")

        self._raw_pointer = pointer
        self.batch = batch

        if hasattr(pointer, "__name__"):
            self.name = pointer.__name__
        else:
            self.name = pointer.__class__.__name__

        self.built = True

    def _build_batcher(self) -> None:
        self._fn = self._raw_pointer if self.batch else torch.vmap(self._raw_pointer)

    @property
    def batch(self) -> bool:
        """Whether the objective accepts a complete population instead of one agent."""

        return self._batch

    @batch.setter
    def batch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("`batch` must be a boolean.")
        self._batch = value
        self._manual = False
        self._build_batcher()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_fn"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_batcher()

    def _evaluate_individually(self, positions: torch.Tensor) -> torch.Tensor:
        results = []
        for position in positions:
            value = self._raw_pointer(position)
            results.append(value if isinstance(value, torch.Tensor) else positions.new_tensor(value))

        return torch.stack(results) if results else positions.new_empty((0,))

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluate one fitness value for each agent.

        Args:
            positions: Population tensor shaped ``(n_agents, n_variables, n_dimensions)``.

        Returns:
            Fitness tensor shaped ``(n_agents,)``.

        Raises:
            TypeError: Positions or batch results are not tensors.
            ValueError: Positions have the wrong rank or results do not contain one scalar per agent.
            ValueError: Eager fitness contains NaN.
            RuntimeError: Compiled fitness contains NaN.

        """

        if not isinstance(positions, torch.Tensor):
            raise TypeError("`positions` must be a torch.Tensor.")
        if positions.ndim != 3:
            raise ValueError(f"`positions` must have three dimensions, but got {positions.ndim}.")

        if self._manual:
            fitness = self._evaluate_individually(positions)
        else:
            try:
                fitness = self._fn(positions)
            except (RuntimeError, ValueError) as error:
                message = str(error)
                vmap_error = message.startswith(("vmap:", "vmap(")) or (
                    message == "DispatchKey FuncTorchBatched doesn't correspond to a device"
                )
                if self.batch or not vmap_error:
                    raise
                logger.warning(f"`pointer={self.name}` cannot use vmap, evaluating agents individually.")
                fitness = self._evaluate_individually(positions)
                self._manual = True

        if not isinstance(fitness, torch.Tensor):
            raise TypeError("`fitness` must be a torch.Tensor.")
        if tuple(fitness.shape) != (positions.shape[0],):
            raise ValueError(f"`fitness` must have shape {(positions.shape[0],)}, but got {tuple(fitness.shape)}.")
        _reject_nan(fitness)

        return fitness

    def __repr__(self) -> str:
        return f"Function(name={self.name}, batch={self.batch})"
