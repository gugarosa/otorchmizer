# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Callback system for optimization lifecycle hooks."""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from otorchmizer.core.function import Function
    from otorchmizer.core.optimizer import UpdateContext
    from otorchmizer.core.population import Population


class Callback:
    """Base callback class defining the optimization lifecycle hooks.

    Notes:
        Subclass and override any hook to inject custom behavior.
        Task start is followed by an initial evaluation and its before/after hooks.
        Each iteration then runs begin, update before/after, evaluation before/after, and end hooks.
        Task end runs after the final iteration.

    """

    def on_task_begin(self, opt_model) -> None:
        """Run after budget validation and before initial evaluation.

        Args:
            opt_model: Live optimization model.

        """

        pass

    def on_task_end(self, opt_model) -> None:
        """Run on normal completion before elapsed history is recorded.

        Args:
            opt_model: Live optimization model.

        """

        pass

    def on_iteration_begin(self, iteration: int, opt_model) -> None:
        """Run before an iteration's update.

        Args:
            iteration: Cumulative one-based iteration count.
            opt_model: Live optimization model.

        """

        pass

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        """Run after evaluation and iteration history recording.

        Args:
            iteration: Cumulative one-based iteration count.
            opt_model: Live optimization model.

        """

        pass

    def on_evaluate_before(self, population: Population, function: Function) -> None:
        """Observe or transform candidates immediately before evaluation.

        Args:
            population: Live population, subject to optimizer-specific mutation restrictions.
            function: Objective adapter at this hook's entry.

        """

        pass

    def on_evaluate_after(self, population: Population, function: Function) -> None:
        """Observe successfully evaluated state.

        Args:
            population: Live evaluated population.
            function: Objective adapter at this hook's entry.

        """

        pass

    def on_update_before(self, ctx: UpdateContext) -> None:
        """Observe state before the driver resolves the update's context.

        Args:
            ctx: Read-only field snapshot referencing live space and objective objects.

        """

        pass

    def on_update_after(self, ctx: UpdateContext) -> None:
        """Observe updated candidates before driver clipping.

        Args:
            ctx: Fresh field snapshot referencing the model's current state.

        """

        pass


class CheckpointCallback(Callback):
    """Periodically saves the optimization model to disk."""

    def __init__(self, file_path: str | PathLike[str] = "checkpoint.pkl", frequency: int = 0) -> None:
        """Configure iteration-based model checkpoints.

        Args:
            file_path: Checkpoint path whose filename receives an iter_<iteration>_ prefix.
            frequency: Nonnegative integer interval, with zero disabling checkpoints.

        Raises:
            TypeError: The path or frequency has an invalid type.
            ValueError: The path is empty or the frequency is negative.

        Notes:
            Parent directories must already exist. Filesystem and serialization errors propagate.

        """

        super().__init__()
        if isinstance(file_path, PathLike):
            file_path = fspath(file_path)
        if not isinstance(file_path, str):
            raise TypeError("`file_path` must be a string or text path-like object.")
        if not file_path or not Path(file_path).name:
            raise ValueError("`file_path` must name a checkpoint file.")
        if not isinstance(frequency, int) or isinstance(frequency, bool):
            raise TypeError("`frequency` must be an integer.")
        if frequency < 0:
            raise ValueError("`frequency` must be non-negative.")
        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        if self.frequency > 0 and iteration % self.frequency == 0:
            path = Path(self.file_path)
            opt_model.save(path.with_name(f"iter_{iteration}_{path.name}"))


class DiscreteSearchCallback(Callback):
    """Maps continuous positions to the nearest allowed discrete values before evaluation."""

    def __init__(self, allowed_values: list[list[int | float]] | None = None) -> None:
        """Retain the discrete values used to snap each variable before evaluation.

        Args:
            allowed_values: One nonempty collection of allowed values per decision variable.

        Notes:
            Task start and evaluation validate finite, nonempty vectors within the population bounds.
            Evaluation compares values in the population's dtype on its device.
            Ties select the first allowed value at the minimum distance.

        """

        super().__init__()
        if allowed_values is not None and not isinstance(allowed_values, list):
            raise TypeError("`allowed_values` must be a list.")
        self.allowed_values = [] if allowed_values is None else allowed_values

    def on_task_begin(self, opt_model) -> None:
        self._values(opt_model.space.population)

    def _values(self, population: Population) -> list[torch.Tensor]:
        n_variables = population.n_variables
        if len(self.allowed_values) != n_variables:
            raise ValueError(f"`allowed_values` must have length {n_variables}, but got {len(self.allowed_values)}.")
        vectors = []
        for i, values in enumerate(self.allowed_values):
            values = torch.as_tensor(values, device=population.device, dtype=population.dtype)
            if values.ndim != 1 or values.numel() == 0:
                raise ValueError("`allowed_values` must contain nonempty vectors.")
            if (
                not torch.isfinite(values).all()
                or not ((values[:, None] >= population.lb[i]) & (values[:, None] <= population.ub[i])).all()
            ):
                raise ValueError("`allowed_values` must contain finite values within every dimension's bounds.")
            vectors.append(values)
        return vectors

    def on_evaluate_before(self, population: Population, function: Function) -> None:
        for i, allowed_t in enumerate(self._values(population)):
            agent_vals = population.positions[:, i, :]
            diffs = torch.abs(agent_vals.unsqueeze(-1) - allowed_t)
            nearest_idx = diffs.argmin(dim=-1)
            population.positions[:, i, :] = allowed_t[nearest_idx]
