# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Hypercomplex mathematical utilities (PyTorch-native)."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import torch

TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def norm(array: torch.Tensor) -> torch.Tensor:
    """Map hypercomplex values to real-valued norms.

    Args:
        array: 2D tensor of shape (n_variables, n_dimensions).

    Returns:
        Norm tensor of shape (n_variables,).

    Notes:
        This is the first step in mapping a hypercomplex number to a real-valued space.

    """

    return torch.linalg.norm(array, dim=1)


def span(
    array: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> torch.Tensor:
    """Map a hypercomplex number between lower and upper bounds.

    Args:
        array: 2D tensor of shape (n_variables, n_dimensions).
        lower_bound: Lower bounds tensor.
        upper_bound: Upper bounds tensor.

    Returns:
        Spanned values usable as decision variables.

    """

    lb = lower_bound.to(array.device)
    ub = upper_bound.to(array.device)

    if lb.dim() == 1:
        lb = lb.unsqueeze(-1)
    if ub.dim() == 1:
        ub = ub.unsqueeze(-1)

    n = norm(array) / (array.shape[1] ** 0.5)
    return (ub.squeeze(-1) - lb.squeeze(-1)) * n + lb.squeeze(-1)


def span_to_hyper_value(
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> Callable[[TensorFunction], TensorFunction]:
    """Create a decorator that maps hypercomplex inputs to real-valued bounds.

    Args:
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Decorator wrapping the objective function.

    """

    def _decorator(f: TensorFunction) -> TensorFunction:
        @wraps(f)
        def _wrapper(x: torch.Tensor) -> torch.Tensor:
            return f(span(x, lb, ub))

        return _wrapper

    return _decorator
