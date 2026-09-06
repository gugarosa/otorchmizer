# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""PyTorch-native distribution generators."""

from __future__ import annotations

from math import gamma, pi, sin, sqrt

import torch


def generate_bernoulli_distribution(
    prob: float = 0.0,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate values from a Bernoulli distribution.

    Args:
        prob: Probability of sampling one.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Binary tensor sampled from the Bernoulli distribution.

    """

    if isinstance(size, int):
        size = (size,)
    return torch.bernoulli(torch.full(size, prob, device=device))


def generate_choice_distribution(
    n: int,
    probs: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Sample indices according to the supplied probability weights.

    Args:
        n: Number of elements to choose from.
        probs: Probability weights on the target device.
        size: Number of samples to draw.

    Returns:
        Selected indices sampled without replacement.

    """

    return torch.multinomial(probs, size, replacement=False)


def generate_levy_distribution(
    beta: float = 0.1,
    size: int | tuple[int, ...] = 1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate values from a Lévy flight distribution.

    Args:
        beta: Symmetric stability exponent in (0, 2].
        size: Shape of the output tensor.
        device: Target device.
        dtype: Sampling dtype, or None to use the PyTorch default.

    Returns:
        Lévy distributed random tensor.

    Raises:
        ValueError: The stability exponent is outside (0, 2].

    Notes:
        Mantegna's ratio sampler is used below two, with the Gaussian limit handled directly at two.
        Unbounded flight steps can exceed the representable range of reduced-precision dtypes.

    References:
        X.-S. Yang and S. Deb. Multiobjective Cuckoo Search for Design Optimization.
        Computers & Operations Research (2013).

    """

    if not 0 < beta <= 2:
        raise ValueError("`beta` must be greater than 0 and at most 2.")
    if isinstance(size, int):
        size = (size,)
    if beta == 2:
        return sqrt(2) * torch.randn(size, device=device, dtype=dtype)

    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)

    u = torch.randn(size, device=device, dtype=dtype) * sigma
    v = torch.randn(size, device=device, dtype=dtype)

    return u / torch.abs(v) ** (1 / beta)
