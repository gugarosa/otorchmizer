"""General-purpose mathematical utilities (PyTorch-native)."""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, List

import torch


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Euclidean distance between two tensors.

    Args:
        x: First tensor.
        y: Second tensor.

    Returns:
        Euclidean distance (scalar or per-sample).
    """

    return torch.linalg.norm(x - y, dim=-1)


def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    """Computes the full pairwise distance matrix for a population.

    Replaces nested O(n²) Python loops (FA, GOA, KH) with a single call.

    Args:
        positions: Tensor of shape (n_agents, n_variables, n_dimensions).

    Returns:
        Distance matrix of shape (n_agents, n_agents).
    """

    flat = positions.reshape(positions.shape[0], -1)
    return torch.cdist(flat, flat)


def kmeans_torch(
    x: torch.Tensor,
    n_clusters: int = 1,
    max_iterations: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """GPU-accelerated K-Means clustering.

    Args:
        x: Input tensor of shape (n_samples, n_variables, n_dimensions).
        n_clusters: Number of clusters.
        max_iterations: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Cluster labels of shape (n_samples,).
    """

    n_samples = x.shape[0]
    flat = x.reshape(n_samples, -1)
    device = x.device

    idx = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = flat[idx].clone()
    labels = torch.zeros(n_samples, dtype=torch.long, device=device)

    for _ in range(max_iterations):
        dists = torch.cdist(flat, centroids)
        new_labels = dists.argmin(dim=1)

        ratio = (new_labels != labels).float().mean()
        if ratio <= tol:
            break

        labels = new_labels

        for i in range(n_clusters):
            mask = labels == i
            if mask.any():
                centroids[i] = flat[mask].mean(dim=0)

    return labels


def n_wise(x: List[Any], size: int = 2) -> Iterable:
    """Iterates over a list and returns n-wise tuples.

    Args:
        x: Values to iterate over.
        size: Tuple size.

    Returns:
        Iterator of n-wise tuples.
    """

    iterator = iter(x)
    return iter(lambda: tuple(islice(iterator, size)), ())


def tournament_selection(
    fitness: torch.Tensor,
    n: int,
    size: int = 2,
) -> torch.Tensor:
    """Vectorized tournament selection.

    Args:
        fitness: Fitness values, shape (n_agents,).
        n: Number of individuals to select.
        size: Tournament size.

    Returns:
        Indices of selected individuals.
    """

    device = fitness.device
    candidates = torch.randint(0, len(fitness), (n, size), device=device)
    candidate_fitness = fitness[candidates]
    winners = candidates[torch.arange(n, device=device), candidate_fitness.argmin(dim=1)]

    return winners


def weighted_wheel_selection(weights: torch.Tensor) -> int:
    """Selects an individual from a weight-based roulette.

    Args:
        weights: Weight values.

    Returns:
        Selected index.
    """

    cumsum = torch.cumsum(weights, dim=0)
    prob = torch.rand(1, device=weights.device) * cumsum[-1]
    idx = (cumsum > prob).nonzero(as_tuple=True)[0]

    return idx[0].item() if len(idx) > 0 else len(weights) - 1
