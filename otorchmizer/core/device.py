# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Centralized device management for the Otorchmizer package.

Supports single-device, multi-GPU, mixed-precision, and CUDA Graph workflows.

"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import torch


class DeviceManager:
    """Manages device resolution and provides factory methods for tensor creation.

    Notes:
        Supports "auto" (picks GPU if available), explicit device strings ("cpu", "cuda:0"),
        and torch.device objects. Provides helpers for multi-GPU, mixed precision, and CUDA Graph capture.

    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Resolve the tensor device and retain the factory dtype.

        Args:
            device: Explicit device or "auto" to prefer the first available CUDA device.
            dtype: Dtype used by tensor factories unless overridden.

        """

        self.device = self._resolve(device)
        self.dtype = dtype

    @staticmethod
    def _resolve(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")

        return torch.device(device)

    @staticmethod
    def available_gpus() -> list[torch.device]:
        """Enumerate available CUDA devices.

        Returns:
            CUDA devices in ascending device-index order.

        """

        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def scatter(tensor: torch.Tensor, devices: Sequence[torch.device]) -> list[torch.Tensor]:
        """Splits a tensor along dim-0 and sends chunks to *devices*.

        Args:
            tensor: Tensor to scatter (split along first dimension).
            devices: Target devices for each chunk.

        Returns:
            List of tensors, one per device, preserving total row count.

        Raises:
            ValueError: No target devices are provided.

        Notes:
            This supports distributing a population across multiple GPUs.
            More devices than rows produces empty chunks for some devices.

        """

        n = len(devices)
        if n == 0:
            raise ValueError("`devices` must contain at least one target.")

        chunks = tensor.tensor_split(n, dim=0)
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

    @contextmanager
    def autocast(self, enabled: bool = True) -> Iterator[None]:
        """Context manager for mixed-precision (float16/bfloat16) computation.

        Args:
            enabled: Whether autocast is active.

        Yields:
            Control within the selected autocast context.

        Notes:
            CPU autocast uses bfloat16. Other devices use the managed dtype if it is float16 or bfloat16,
            otherwise float16. PyTorch selects which operations to downcast and which retain higher precision.
            Disabling this context leaves any enclosing autocast context unchanged.

        Examples:
            Enable mixed precision for supported operations::

                dm = DeviceManager("cuda:0", dtype=torch.float16)
                with dm.autocast():
                    result = some_tensor_operation(...)

        """

        if not enabled:
            yield
            return

        if self.device.type == "cpu":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16) else torch.float16

        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=enabled):
            yield

    @staticmethod
    def supports_cuda_graphs() -> bool:
        """Returns True if the current environment supports CUDA Graphs."""

        return torch.cuda.is_available()

    @staticmethod
    def capture_graph(callable_fn: Callable, *static_args, warmup: int = 3) -> CUDAGraphRunner:
        """Captures a CUDA Graph from *callable_fn* for replay.

        Args:
            callable_fn: A callable that operates on *static_args* in-place.
            *static_args: Pre-allocated tensors that will be reused across replays.
            warmup: Number of warm-up runs before capture.

        Returns:
            A :class:`CUDAGraphRunner` instance with a ``replay()`` method.

        Raises:
            RuntimeError: If CUDA is not available.

        Notes:
            CUDA Graphs record a fixed sequence of GPU operations for replay in a single submission.
            This reduces Python and kernel-launch overhead for update loops with fixed-shape tensors.
            Warm-up and capture both invoke the callable, so they can mutate the supplied arguments.

        """

        if not torch.cuda.is_available():
            raise RuntimeError("`capture_graph` requires a CUDA-capable device.")

        # Warmup to trigger lazy initializations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                callable_fn(*static_args)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            callable_fn(*static_args)

        return CUDAGraphRunner(g, static_args)

    def zeros(self, *shape, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Create a zero-filled tensor on the managed device.

        Args:
            *shape: Tensor dimensions accepted by torch.zeros.
            dtype: Dtype override, or None to use the managed dtype.

        Returns:
            Zero-filled tensor with the requested shape and dtype.

        """

        return torch.zeros(*shape, dtype=dtype or self.dtype, device=self.device)

    def ones(self, *shape, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Create a ones-filled tensor on the managed device.

        Args:
            *shape: Tensor dimensions accepted by torch.ones.
            dtype: Dtype override, or None to use the managed dtype.

        Returns:
            Ones-filled tensor with the requested shape and dtype.

        """

        return torch.ones(*shape, dtype=dtype or self.dtype, device=self.device)

    def rand(self, *shape) -> torch.Tensor:
        """Sample a uniform random tensor on the managed device.

        Args:
            *shape: Tensor dimensions accepted by torch.rand.

        Returns:
            Tensor sampled from [0, 1) using the managed dtype.

        """

        return torch.rand(*shape, device=self.device, dtype=self.dtype)

    def randn(self, *shape) -> torch.Tensor:
        """Sample a standard normal tensor on the managed device.

        Args:
            *shape: Tensor dimensions accepted by torch.randn.

        Returns:
            Standard normal samples using the managed dtype.

        """

        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    def full(self, shape: tuple, fill_value: float, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Create a constant-filled tensor on the managed device.

        Args:
            shape: Output tensor dimensions.
            fill_value: Value assigned to every element.
            dtype: Dtype override, or None to use the managed dtype.

        Returns:
            Constant-filled tensor with the requested shape and dtype.

        """

        return torch.full(shape, fill_value, dtype=dtype or self.dtype, device=self.device)

    def tensor(self, data, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Copy data into a tensor on the managed device.

        Args:
            data: Data accepted by torch.tensor.
            dtype: Dtype override, or None to use the managed dtype.

        Returns:
            Tensor containing a copy of the supplied data.

        """

        return torch.tensor(data, dtype=dtype or self.dtype, device=self.device)


class CUDAGraphRunner:
    """Thin wrapper around a captured ``torch.cuda.CUDAGraph``.

    Examples:
        Replay a captured update after its warm-up and capture calls::

            runner = DeviceManager.capture_graph(my_update_fn, pos, vel)
            for _ in range(n_iterations):
                runner.replay()

    """

    def __init__(self, graph: torch.cuda.CUDAGraph, static_args: tuple) -> None:
        """Retain a captured graph and its static argument references.

        Args:
            graph: CUDA Graph whose operations are replayed.
            static_args: Arguments kept alive for the captured operations.

        """

        self.graph = graph
        self.static_args = static_args

    def replay(self) -> None:
        """Replays the captured graph."""

        self.graph.replay()
