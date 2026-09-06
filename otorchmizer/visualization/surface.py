# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""3-D benchmarking function surface plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def plot(
    points: Sequence[torch.Tensor | np.ndarray],
    title: str = "",
    subtitle: str = "",
    style: str = "winter",
    colorbar: bool = True,
) -> None:
    """Plot a 3-D surface from function evaluation points.

    Args:
        points: Three arrays or tensors containing the X, Y, and Z coordinates.
        title: Plot title.
        subtitle: Plot subtitle.
        style: Surface colormap style.
        colorbar: Whether to display colorbar.

    Notes:
        This function displays the pyplot window and returns None.

    """

    points = [_to_numpy(p) for p in points]

    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection="3d")

    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$f(x, y)$")
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")
    ax.tick_params(labelsize=8)

    ax.plot_wireframe(points[0], points[1], points[2], color="grey")
    surface = ax.plot_surface(
        points[0],
        points[1],
        points[2],
        rstride=1,
        cstride=1,
        cmap=style,
        edgecolor="none",
    )

    if colorbar:
        fig.colorbar(surface, shrink=0.5, aspect=10)

    plt.show()
