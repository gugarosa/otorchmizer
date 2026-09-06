# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""History tracking for optimization runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import otorchmizer.utils.exception as e


class History:
    """Records per-iteration optimization data.

    Notes:
        Uses dump() to append arbitrary key-value pairs to per-key histories.
        Direct tensor values and the best-agent tensor pair are detached and moved to CPU before storage
        to avoid retaining GPU tensor storage.

    """

    def __init__(self, save_agents: bool = False) -> None:
        """Configure whether population positions are retained in history.

        Args:
            save_agents: Whether to save all agent positions each iteration.

        Raises:
            TypeError: The save-agents flag is not a boolean.

        """

        self.save_agents = save_agents

    @property
    def save_agents(self) -> bool:
        """Whether dump retains the positions key instead of skipping it."""

        return self._save_agents

    @save_agents.setter
    def save_agents(self, save_agents: bool) -> None:
        if not isinstance(save_agents, bool):
            raise e.TypeError("`save_agents` should be a boolean.")
        self._save_agents = save_agents

    @staticmethod
    def _to_python(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    def _parse(self, key: str, value: Any) -> Any:
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

        Args:
            **kwargs: Named values appended to their corresponding history lists.

        Notes:
            Each key becomes a list attribute and receives one entry per call supplying that key.
            The positions key is skipped unless save_agents is enabled.

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

    def get_convergence(self, key: str, index: int = 0) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Gets the convergence list of a specified key.

        Args:
            key: Key to retrieve.
            index: Compatibility argument retained without affecting retrieval.

        Returns:
            History as an array, or a pair of position and fitness arrays for best_agent.

        Raises:
            AttributeError: No history attribute exists for the requested key.

        """

        attr = np.asarray(getattr(self, key), dtype=object)

        if key == "best_agent":
            positions = [a[0] for a in attr]
            fitnesses = [a[1] for a in attr]
            return np.array(positions), np.array(fitnesses)

        return np.array(attr.tolist())
