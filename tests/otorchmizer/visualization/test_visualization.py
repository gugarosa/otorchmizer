"""Tests for visualization modules (convergence and surface).

Matplotlib is configured to use 'Agg' backend for headless testing.
"""

import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from otorchmizer.visualization import convergence, surface
import otorchmizer.utils.exception as e


class TestConvergencePlot:
    def teardown_method(self):
        plt.close("all")

    def test_single_variable(self):
        data = [1.0, 0.8, 0.6, 0.4, 0.2]
        convergence.plot(data, labels=["loss"], title="Test")

    def test_multiple_variables(self):
        data1 = list(range(10))
        data2 = list(range(10, 0, -1))
        convergence.plot(data1, data2, labels=["a", "b"])

    def test_tensor_input(self):
        data = torch.linspace(0, 1, 20)
        convergence.plot(data, labels=["tensor_data"])

    def test_numpy_input(self):
        data = np.linspace(0, 1, 20)
        convergence.plot(data, labels=["np_data"])

    def test_no_labels_auto_generated(self):
        data = [1, 2, 3]
        convergence.plot(data)

    def test_no_grid(self):
        convergence.plot([1, 2, 3], grid=False)

    def test_no_legend(self):
        convergence.plot([1, 2, 3], legend=False)

    def test_custom_labels(self):
        convergence.plot([1, 2], [3, 4], labels=["x", "y"],
                        title="Title", subtitle="Sub",
                        xlabel="step", ylabel="val")

    def test_invalid_labels_type(self):
        with pytest.raises(e.TypeError):
            convergence.plot([1, 2, 3], labels="not_a_list")

    def test_labels_size_mismatch(self):
        with pytest.raises(e.SizeError):
            convergence.plot([1, 2], [3, 4], labels=["only_one"])


class TestSurfacePlot:
    def teardown_method(self):
        plt.close("all")

    def test_numpy_input(self):
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = X ** 2 + Y ** 2
        surface.plot([X, Y, Z], title="Sphere")

    def test_tensor_input(self):
        x = torch.linspace(-1, 1, 10)
        y = torch.linspace(-1, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = X ** 2 + Y ** 2
        surface.plot([X, Y, Z], title="Sphere Tensor")

    def test_no_colorbar(self):
        x = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, x)
        Z = X + Y
        surface.plot([X, Y, Z], colorbar=False)

    def test_custom_style(self):
        x = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, x)
        Z = X * Y
        surface.plot([X, Y, Z], style="viridis")
