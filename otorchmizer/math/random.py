# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""PyTorch-native random number generators.

All generators support batched creation on any device (CPU/GPU).

"""

from __future__ import annotations

from math import prod
from operator import index

import torch


def generate_binary_random_number(
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate binary random values by rounding uniform samples.

    Args:
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Binary random tensor with values in {0, 1}.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.round(torch.rand(size, device=device))


def generate_exponential_random_number(
    scale: float = 1.0,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate values from an exponential distribution.

    Args:
        scale: Scale of the distribution, equal to the reciprocal rate.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Exponentially distributed random tensor.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.distributions.Exponential(1.0 / scale).sample(size).to(device)


def generate_gamma_random_number(
    shape: float = 1.0,
    scale: float = 1.0,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate values from a gamma distribution.

    Args:
        shape: Shape parameter.
        scale: Scale parameter.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Gamma distributed random tensor.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.distributions.Gamma(shape, 1.0 / scale).sample(size).to(device)


def generate_integer_random_number(
    low: int = 0,
    high: int = 1,
    exclude_value: int | None = None,
    size: int | tuple[int, ...] | None = None,
    device: torch.device = torch.device("cpu"),
) -> int | torch.Tensor:
    """Generate random integers in the half-open interval [low, high).

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        exclude_value: Integer value to exclude from the sampling range.
        size: Shape of output. If None, returns a Python int.
        device: Target device.

    Returns:
        Random integer or tensor of integers.

    Raises:
        ValueError: If excluding the only possible integer would produce a nonempty result.

    """

    scalar = size is None
    shape = (1,) if scalar else (size,) if isinstance(size, int) else size
    if exclude_value is not None:
        exclude_value = index(exclude_value)

    if exclude_value is not None and low <= exclude_value < high:
        if high - low == 1:
            if prod(shape) != 0:
                raise ValueError("`exclude_value` must leave at least one possible integer.")
            result = torch.empty(shape, dtype=torch.long, device=device)
        else:
            # Shift a uniform draw around the excluded integer instead of retrying whole tensors
            result = torch.randint(low, high - 1, shape, device=device)
            result += (result >= exclude_value).long()
    else:
        result = torch.randint(low, high, shape, device=device)

    return result.item() if scalar else result


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate values from a uniform distribution in [low, high).

    Args:
        low: Lower bound.
        high: Upper bound.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Uniformly distributed random tensor.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.rand(size, device=device) * (high - low) + low


def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate values from a Gaussian distribution.

    Args:
        mean: Mean of the distribution.
        variance: Standard deviation.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Normally distributed random tensor.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.randn(size, device=device) * variance + mean
