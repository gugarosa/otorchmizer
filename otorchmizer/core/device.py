"""Centralized device management for the Otorchmizer package.

Supports single-device, multi-GPU, mixed-precision, and CUDA Graph workflows.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Sequence

import torch


class DeviceManager:
    """Manages device resolution and provides factory methods for tensor creation.

    Supports "auto" (picks GPU if available), explicit device strings ("cpu", "cuda:0"),
    and torch.device objects.  Also provides helpers for multi-GPU, mixed-precision,
    and CUDA Graph capture.
    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = self._resolve(device)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(device: str | torch.device) -> torch.device:
        """Resolves a device specifier to a torch.device."""

        if isinstance(device, torch.device):
            return device

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")

        return torch.device(device)

    # ------------------------------------------------------------------
    # Multi-GPU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def available_gpus() -> List[torch.device]:
        """Returns a list of all available CUDA devices."""

        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def scatter(tensor: torch.Tensor, devices: Sequence[torch.device]) -> List[torch.Tensor]:
        """Splits a tensor along dim-0 and sends chunks to *devices*.

        Useful for distributing a population across multiple GPUs.

        Args:
            tensor: Tensor to scatter (split along first dimension).
            devices: Target devices for each chunk.

        Returns:
            List of tensors, one per device, preserving total row count.
        """

        n = len(devices)
        chunks = tensor.chunk(n, dim=0)
        return [ch.to(dev) for ch, dev in zip(chunks, devices)]

    @staticmethod
    def gather(tensors: Sequence[torch.Tensor], target_device: torch.device) -> torch.Tensor:
        """Concatenates tensors from different devices onto *target_device*.

        Args:
            tensors: Sequence of tensors (potentially on different devices).
            target_device: Device for the gathered result.

        Returns:
            Concatenated tensor on *target_device*.
        """

        return torch.cat([t.to(target_device) for t in tensors], dim=0)

    # ------------------------------------------------------------------
    # Mixed-precision helpers
    # ------------------------------------------------------------------

    @contextmanager
    def autocast(self, enabled: bool = True):
        """Context manager for mixed-precision (float16/bfloat16) computation.

        On CUDA devices, uses torch.amp.autocast for automatic downcasting
        of eligible operations (matmuls, convolutions, etc.) while keeping
        reductions in float32 for numerical stability.

        On CPU, uses bfloat16 autocast when available (PyTorch ≥ 2.0).

        Usage::

            dm = DeviceManager("cuda:0", dtype=torch.float16)
            with dm.autocast():
                result = some_tensor_operation(...)

        Args:
            enabled: Whether autocast is active.  Pass ``False`` to no-op.
        """

        if not enabled or self.device.type == "cpu":
            yield
            return

        amp_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.float16
        with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=enabled):
            yield

    # ------------------------------------------------------------------
    # CUDA Graph helpers
    # ------------------------------------------------------------------

    @staticmethod
    def supports_cuda_graphs() -> bool:
        """Returns True if the current environment supports CUDA Graphs."""

        return torch.cuda.is_available()

    @staticmethod
    def capture_graph(callable_fn, *static_args, warmup: int = 3):
        """Captures a CUDA Graph from *callable_fn* for replay.

        CUDA Graphs eliminate Python and kernel-launch overhead by recording
        a fixed sequence of GPU operations and replaying them in a single
        submission.  Best for update loops with fixed-shape tensors.

        Args:
            callable_fn: A callable that operates on *static_args* in-place.
            *static_args: Pre-allocated tensors that will be reused across replays.
            warmup: Number of warm-up runs before capture (default 3).

        Returns:
            A :class:`CUDAGraphRunner` instance with a ``replay()`` method.

        Raises:
            RuntimeError: If CUDA is not available.
        """

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graphs require a CUDA-capable device")

        # Warmup to trigger lazy initializations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                callable_fn(*static_args)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            callable_fn(*static_args)

        return CUDAGraphRunner(g, static_args)

    # ------------------------------------------------------------------
    # Tensor factory helpers
    # ------------------------------------------------------------------

    def zeros(self, *shape, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Creates a zero-filled tensor on the managed device."""

        return torch.zeros(*shape, dtype=dtype or self.dtype, device=self.device)

    def ones(self, *shape, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Creates a ones-filled tensor on the managed device."""

        return torch.ones(*shape, dtype=dtype or self.dtype, device=self.device)

    def rand(self, *shape) -> torch.Tensor:
        """Creates a uniform random tensor on the managed device."""

        return torch.rand(*shape, device=self.device)

    def randn(self, *shape) -> torch.Tensor:
        """Creates a normal random tensor on the managed device."""

        return torch.randn(*shape, device=self.device)

    def full(self, shape: tuple, fill_value: float,
             dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Creates a tensor filled with a constant on the managed device."""

        return torch.full(shape, fill_value, dtype=dtype or self.dtype, device=self.device)

    def tensor(self, data, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Creates a tensor from data on the managed device."""

        return torch.tensor(data, dtype=dtype or self.dtype, device=self.device)


class CUDAGraphRunner:
    """Thin wrapper around a captured ``torch.cuda.CUDAGraph``.

    Usage::

        runner = DeviceManager.capture_graph(my_update_fn, pos, vel)
        for _ in range(n_iterations):
            runner.replay()   # near-zero Python overhead
    """

    def __init__(self, graph: torch.cuda.CUDAGraph, static_args: tuple) -> None:
        self.graph = graph
        self.static_args = static_args

    def replay(self) -> None:
        """Replays the captured graph."""

        self.graph.replay()
