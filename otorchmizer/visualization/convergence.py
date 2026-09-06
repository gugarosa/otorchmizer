# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Convergence plots with tensor-to-numpy bridge."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

PlotData = torch.Tensor | np.ndarray | Sequence[float]


def _to_numpy(arg: PlotData) -> np.ndarray | Sequence[float]:
    if isinstance(arg, torch.Tensor):
        return arg.detach().cpu().numpy()
    return arg


def plot(
    *args: PlotData,
    labels: list[str] | None = None,
    title: str = "",
    subtitle: str = "",
    xlabel: str = "iteration",
    ylabel: str = "value",
    grid: bool = True,
    legend: bool = True,
) -> None:
    """Plot convergence graphs for one or more variables.

    Args:
        *args: Lists, NumPy arrays, or tensors containing one value per iteration.
        labels: Labels for each plot line.
        title: Plot title.
        subtitle: Plot subtitle.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        grid: Whether to display grid lines.
        legend: Whether to display legend.

    Raises:
        TypeError: If `labels` is not a list.
        ValueError: If `labels` does not contain one entry per plotted variable.

    Notes:
        This function displays the pyplot window and returns None.

    """

    _, ax = plt.subplots(figsize=(7, 5))

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    if labels:
        if not isinstance(labels, list):
            raise TypeError("`labels` should be a list.")
        if len(labels) != len(args):
            raise ValueError("`args` and `labels` should have the same size.")
    else:
        labels = [f"variable_{i}" for i in range(len(args))]

    for arg, label in zip(args, labels):
        ax.plot(_to_numpy(arg), label=label)

    if legend:
        ax.legend()

    plt.show()
