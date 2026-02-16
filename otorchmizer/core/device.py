"""Centralized device management for the Otorchmizer package."""

from __future__ import annotations

import torch


class DeviceManager:
    """Manages device resolution and provides factory methods for tensor creation.

    Supports "auto" (picks GPU if available), explicit device strings ("cpu", "cuda:0"),
    and torch.device objects.
    """

    def __init__(self, device: str | torch.device = "auto") -> None:
        self.device = self._resolve(device)

    @staticmethod
    def _resolve(device: str | torch.device) -> torch.device:
        """Resolves a device specifier to a torch.device.

        Args:
            device: "auto", a device string, or a torch.device.

        Returns:
            Resolved torch.device.
        """

        if isinstance(device, torch.device):
            return device

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")

        return torch.device(device)

    def zeros(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Creates a zero-filled tensor on the managed device."""

        return torch.zeros(*shape, dtype=dtype, device=self.device)

    def ones(self, *shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Creates a ones-filled tensor on the managed device."""

        return torch.ones(*shape, dtype=dtype, device=self.device)

    def rand(self, *shape) -> torch.Tensor:
        """Creates a uniform random tensor on the managed device."""

        return torch.rand(*shape, device=self.device)

    def randn(self, *shape) -> torch.Tensor:
        """Creates a normal random tensor on the managed device."""

        return torch.randn(*shape, device=self.device)

    def full(self, shape: tuple, fill_value: float,
             dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Creates a tensor filled with a constant on the managed device."""

        return torch.full(shape, fill_value, dtype=dtype, device=self.device)

    def tensor(self, data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Creates a tensor from data on the managed device."""

        return torch.tensor(data, dtype=dtype, device=self.device)
