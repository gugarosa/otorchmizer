# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""History tracking for optimization runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


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
            raise TypeError("`save_agents` should be a boolean.")
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

    def get_convergence(self, key: str, index: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Gets the convergence list of a specified key.

        Args:
            key: Key to retrieve.
            index: Agent index for positions or fitness, or None to return all recorded agents.

        Returns:
            History as an array, or a pair of position and fitness arrays for best_agent.

        Raises:
            AttributeError: No history attribute exists for the requested key.
            TypeError: The agent index is not an integer or None.
            ValueError: An index is supplied for a key without an agent axis.
            IndexError: The index is outside the recorded population.

        Notes:
            Positions have shape (n_records, n_agents, n_variables, n_dimensions) and fitness (n_records, n_agents).
            Agent selection removes only its axis. Best positions retain (n_records, n_variables, n_dimensions).
            With changing population sizes, selecting an index stacks that row from each record.
            Unindexed ragged positions or fitness return a one-dimensional object array of per-record arrays.

        """

        if index is not None and key not in ("positions", "fitness"):
            raise ValueError("`index` is only supported for positions and fitness histories.")
        if index is not None and (not isinstance(index, (int, np.integer)) or isinstance(index, bool)):
            raise TypeError("`index` must be an integer or None.")
        attr = getattr(self, key)

        if key == "best_agent":
            positions = [a[0] for a in attr]
            fitnesses = [a[1] for a in attr]
            return np.array(positions), np.array(fitnesses)

        if index is not None:
            return np.asarray([record[index] for record in attr])
        if key in ("positions", "fitness") and len({np.shape(record) for record in attr}) > 1:
            result = np.empty(len(attr), dtype=object)
            for i, record in enumerate(attr):
                result[i] = np.asarray(record)
            return result
        return np.asarray(attr)
