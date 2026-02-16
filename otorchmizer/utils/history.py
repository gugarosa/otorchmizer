"""History tracking for optimization runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import otorchmizer.utils.exception as e


class History:
    """Records per-iteration optimization data.

    Uses dump() to store arbitrary key-value pairs per iteration.
    Tensors are automatically detached and moved to CPU before storage
    to prevent GPU memory leaks.
    """

    def __init__(self, save_agents: bool = False) -> None:
        """Initialization method.

        Args:
            save_agents: Whether to save all agent positions each iteration.
        """

        self.save_agents = save_agents

    @property
    def save_agents(self) -> bool:
        return self._save_agents

    @save_agents.setter
    def save_agents(self, save_agents: bool) -> None:
        if not isinstance(save_agents, bool):
            raise e.TypeError("`save_agents` should be a boolean")
        self._save_agents = save_agents

    @staticmethod
    def _to_python(value: Any) -> Any:
        """Converts tensors to Python-native types for storage."""

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    def _parse(self, key: str, value: Any) -> Any:
        """Parses incoming values based on key.

        Args:
            key: Data key.
            value: Data value (may contain tensors).

        Returns:
            Parsed value safe for CPU storage.
        """

        if key == "best_agent":
            pos, fit = value
            return (self._to_python(pos), self._to_python(fit))

        if key == "positions":
            return self._to_python(value)

        if key == "fitness":
            return self._to_python(value)

        return value

    def dump(self, **kwargs) -> None:
        """Dumps key-value pairs into the history.

        Each key becomes a list attribute, appended per iteration.
        """

        for key, value in kwargs.items():
            if key == "positions" and not self.save_agents:
                continue

            if key in ("best_agent", "positions", "fitness"):
                output = self._parse(key, value)
            else:
                output = self._to_python(value) if isinstance(value, torch.Tensor) else value

            if not hasattr(self, key):
                setattr(self, key, [output])
            else:
                getattr(self, key).append(output)

    def get_convergence(self, key: str, index: int = 0) -> np.ndarray:
        """Gets the convergence list of a specified key.

        Args:
            key: Key to retrieve.
            index: Index for per-agent retrieval.

        Returns:
            Values as a numpy array.
        """

        attr = np.asarray(getattr(self, key), dtype=object)

        if key == "best_agent":
            positions = [a[0] for a in attr]
            fitnesses = [a[1] for a in attr]
            return np.array(positions), np.array(fitnesses)

        return np.array(attr.tolist())
