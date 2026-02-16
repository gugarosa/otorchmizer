"""Hypercomplex mathematical utilities (PyTorch-native)."""

from __future__ import annotations

from functools import wraps
from typing import Union

import torch


def norm(array: torch.Tensor) -> torch.Tensor:
    """Calculates the norm over the dimension axis.

    Maps a hypercomplex number to a real-valued space (first step).

    Args:
        array: 2D tensor of shape (n_variables, n_dimensions).

    Returns:
        Norm tensor of shape (n_variables,).
    """

    return torch.linalg.norm(array, dim=1)


def span(
    array: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> torch.Tensor:
    """Spans a hypercomplex number between lower and upper bounds.

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
) -> callable:
    """Decorator that spans hypercomplex inputs to real-valued bounds.

    Args:
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Decorator wrapping the objective function.
    """

    def _decorator(f: callable) -> callable:
        @wraps(f)
        def _wrapper(x: torch.Tensor) -> torch.Tensor:
            return f(span(x, lb, ub))
        return _wrapper

    return _decorator
