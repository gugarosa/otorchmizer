# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Callback system for optimization lifecycle hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import otorchmizer.utils.exception as e

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
        pass

    def on_task_end(self, opt_model) -> None:
        pass

    def on_iteration_begin(self, iteration: int, opt_model) -> None:
        pass

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        pass

    def on_evaluate_before(self, population: Population, function: Function) -> None:
        pass

    def on_evaluate_after(self, population: Population, function: Function) -> None:
        pass

    def on_update_before(self, ctx: UpdateContext) -> None:
        pass

    def on_update_after(self, ctx: UpdateContext) -> None:
        pass


class CallbackVessel:
    """Aggregates multiple callbacks and dispatches events to all of them."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        """Retain callbacks in their dispatch order.

        Args:
            callbacks: Callback instances receiving each lifecycle event.

        """

        self.callbacks = callbacks or []

    @property
    def callbacks(self) -> list[Callback]:
        """Mutable callback list traversed in order for each lifecycle event."""

        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: list[Callback]) -> None:
        if not isinstance(callbacks, list):
            raise e.TypeError("`callbacks` should be a list.")
        self._callbacks = callbacks

    def on_task_begin(self, opt_model) -> None:
        for cb in self.callbacks:
            cb.on_task_begin(opt_model)

    def on_task_end(self, opt_model) -> None:
        for cb in self.callbacks:
            cb.on_task_end(opt_model)

    def on_iteration_begin(self, iteration: int, opt_model) -> None:
        for cb in self.callbacks:
            cb.on_iteration_begin(iteration, opt_model)

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        for cb in self.callbacks:
            cb.on_iteration_end(iteration, opt_model)

    def on_evaluate_before(self, population: Population, function: Function) -> None:
        for cb in self.callbacks:
            cb.on_evaluate_before(population, function)

    def on_evaluate_after(self, population: Population, function: Function) -> None:
        for cb in self.callbacks:
            cb.on_evaluate_after(population, function)

    def on_update_before(self, ctx: UpdateContext) -> None:
        for cb in self.callbacks:
            cb.on_update_before(ctx)

    def on_update_after(self, ctx: UpdateContext) -> None:
        for cb in self.callbacks:
            cb.on_update_after(ctx)


class CheckpointCallback(Callback):
    """Periodically saves the optimization model to disk."""

    def __init__(self, file_path: str = "checkpoint.pkl", frequency: int = 0) -> None:
        """Configure iteration-based model checkpoints.

        Args:
            file_path: Checkpoint filename prefixed with iter_<iteration>_ when saved.
            frequency: Save interval in iterations, with nonpositive values disabling checkpoints.

        """

        super().__init__()
        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        if self.frequency > 0 and iteration % self.frequency == 0:
            opt_model.save(f"iter_{iteration}_{self.file_path}")


class DiscreteSearchCallback(Callback):
    """Maps continuous positions to the nearest allowed discrete values before evaluation."""

    def __init__(self, allowed_values: list[list[int | float]] | None = None) -> None:
        """Retain the discrete values used to snap each variable before evaluation.

        Args:
            allowed_values: One nonempty collection of allowed values per decision variable.

        Notes:
            Task start validates the number of collections and rejects empty collections.
            Evaluation compares values in the population's dtype on its device.
            Ties select the first allowed value at the minimum distance.

        """

        super().__init__()
        self.allowed_values = allowed_values or []

    def on_task_begin(self, opt_model) -> None:
        n_variables = opt_model.space.population.n_variables
        if len(self.allowed_values) != n_variables:
            raise e.SizeError(f"`allowed_values` must have length {n_variables}, but got {len(self.allowed_values)}.")
        if any(not values for values in self.allowed_values):
            raise e.ValueError("`allowed_values` must contain a nonempty collection for each variable.")

    def on_evaluate_before(self, population: Population, function: Function) -> None:
        for i, allowed in enumerate(self.allowed_values):
            allowed_t = torch.tensor(allowed, device=population.device, dtype=population.dtype)
            agent_vals = population.positions[:, i, :]
            diffs = torch.abs(agent_vals.unsqueeze(-1) - allowed_t)
            nearest_idx = diffs.argmin(dim=-1)
            population.positions[:, i, :] = allowed_t[nearest_idx]
