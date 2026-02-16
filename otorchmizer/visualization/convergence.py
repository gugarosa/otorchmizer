"""Convergence plots with tensor-to-numpy bridge."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import torch

import otorchmizer.utils.exception as e


def _to_numpy(arg):
    """Converts a tensor or array to numpy for matplotlib."""

    if isinstance(arg, torch.Tensor):
        return arg.detach().cpu().numpy()
    return arg


def plot(
    *args,
    labels: Optional[List[str]] = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "iteration",
    ylabel: str = "value",
    grid: bool = True,
    legend: bool = True,
) -> None:
    """Plots convergence graphs of desired variables.

    Each variable is a list, numpy array, or tensor with size n_iterations.

    Args:
        labels: Labels for each plot line.
        title: Plot title.
        subtitle: Plot subtitle.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        grid: Whether to display grid lines.
        legend: Whether to display legend.
    """

    _, ax = plt.subplots(figsize=(7, 5))

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    if labels:
        if not isinstance(labels, list):
            raise e.TypeError("`labels` should be a list")
        if len(labels) != len(args):
            raise e.SizeError("`args` and `labels` should have the same size")
    else:
        labels = [f"variable_{i}" for i in range(len(args))]

    for arg, label in zip(args, labels):
        ax.plot(_to_numpy(arg), label=label)

    if legend:
        ax.legend()

    plt.show()
