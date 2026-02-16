"""PyTorch-native random number generators.

All generators support batched creation on any device (CPU/GPU).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def generate_binary_random_number(
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates binary random values based on uniform rounding.

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
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates random values from an exponential distribution.

    Args:
        scale: Scale (1/rate) of the distribution.
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
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates random values from a gamma distribution.

    Args:
        shape: Shape parameter (alpha).
        scale: Scale parameter (beta).
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
    exclude_value: Optional[int] = None,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    device: torch.device = torch.device("cpu"),
) -> Union[int, torch.Tensor]:
    """Generates random integers in [low, high).

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (exclusive).
        exclude_value: Value to exclude (retries if generated).
        size: Shape of output. If None, returns a Python int.
        device: Target device.

    Returns:
        Random integer or tensor of integers.
    """

    if size is None:
        val = torch.randint(low, high, (1,), device=device).item()
        if exclude_value is not None and val == exclude_value:
            return generate_integer_random_number(low, high, exclude_value, size, device)
        return val

    if isinstance(size, int):
        size = (size,)

    result = torch.randint(low, high, size, device=device)

    if exclude_value is not None and (result == exclude_value).any():
        return generate_integer_random_number(low, high, exclude_value, size, device)

    return result


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates random values from a uniform distribution in [low, high).

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
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates random values from a gaussian (normal) distribution.

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
