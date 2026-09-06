# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Weighted multi-objective function wrapper."""

from __future__ import annotations

from collections.abc import Callable

import torch

from otorchmizer.core.function import _reject_nan
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """Scalarizes multiple objectives via weighted sum.

    Notes:
        Returns one fitness value per agent using z = Σ(wᵢ · fᵢ(x)).
        Weights retain their Python precision and are converted to the objective's device and dtype at evaluation.

    """

    def __init__(self, functions: list[Callable], weights: list[float], batch: bool = False) -> None:
        """Wrap objectives and retain one scalarization weight per objective.

        Args:
            functions: List of objective callables.
            weights: Per-objective weights for scalarization.
            batch: If True, callables handle full population tensors.

        Raises:
            TypeError: Functions or weights are not lists, or an objective is not callable.
            ValueError: The number of weights differs from the number of objectives.

        """

        super().__init__(functions, batch)

        if not isinstance(weights, list):
            raise TypeError("`weights` should be a list.")
        if len(weights) != len(self.functions):
            raise ValueError("`weights` should have the same size as `functions`.")

        self.weights = weights

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates weighted sum of objectives.

        Args:
            positions: Population tensor shaped (n_agents, n_variables, n_dimensions).

        Returns:
            Scalarized fitness tensor of shape (n_agents,).

        Raises:
            ValueError: Eager objective or scalarized fitness contains NaN.
            RuntimeError: Compiled objective or scalarized fitness contains NaN.

        """

        objectives = super().__call__(positions)
        if len(self.weights) != len(self.functions):
            raise ValueError("`weights` must have the same size as `functions`.")
        dtype = torch.promote_types(objectives.dtype, positions.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        w = torch.as_tensor(self.weights, device=objectives.device, dtype=dtype)
        if w.ndim != 1:
            raise ValueError("`weights` must be a vector of scalar values.")
        result = (objectives * w).sum(dim=-1)
        _reject_nan(result)
        return result
