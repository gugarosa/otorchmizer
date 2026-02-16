"""PyTorch-native distribution generators."""

from __future__ import annotations

from math import gamma, pi, sin
from typing import Tuple, Union

import torch


def generate_bernoulli_distribution(
    prob: float = 0.0,
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a Bernoulli distribution.

    Args:
        prob: Probability of 1.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Binary tensor sampled from Bernoulli(prob).
    """

    if isinstance(size, int):
        size = (size,)
    return torch.bernoulli(torch.full(size, prob, device=device))


def generate_choice_distribution(
    n: int,
    probs: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Generates a random choice distribution based on probabilities.

    Args:
        n: Number of elements to choose from.
        probs: Probability weights (must be on correct device already).
        size: Number of samples to draw.

    Returns:
        Tensor of selected indices (without replacement).
    """

    return torch.multinomial(probs, size, replacement=False)


def generate_levy_distribution(
    beta: float = 0.1,
    size: Union[int, Tuple[int, ...]] = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a Lévy flight distribution.

    References:
        X.-S. Yang and S. Deb. Multiobjective Cuckoo Search for Design Optimization.
        Computers & Operations Research (2013).

    Args:
        beta: Skewness parameter.
        size: Shape of the output tensor.
        device: Target device.

    Returns:
        Lévy distributed random tensor.
    """

    if isinstance(size, int):
        size = (size,)

    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)

    u = torch.randn(size, device=device) * sigma
    v = torch.randn(size, device=device)

    return u / torch.abs(v) ** (1 / beta)
