"""Callback system for optimization lifecycle hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import torch

import otorchmizer.utils.exception as e

if TYPE_CHECKING:
    from otorchmizer.core.optimizer import UpdateContext
    from otorchmizer.core.population import Population
    from otorchmizer.core.function import Function


class Callback:
    """Base callback class defining the optimization lifecycle hooks.

    Subclass and override any method to inject custom behavior.

    Lifecycle:
        on_task_begin → [on_iteration_begin → on_update_before → UPDATE →
        on_update_after → on_evaluate_before → EVALUATE → on_evaluate_after →
        on_iteration_end] × N → on_task_end
    """

    def on_task_begin(self, opt_model) -> None:
        pass

    def on_task_end(self, opt_model) -> None:
        pass

    def on_iteration_begin(self, iteration: int, opt_model) -> None:
        pass

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        pass

    def on_evaluate_before(self, population: "Population", function: "Function") -> None:
        pass

    def on_evaluate_after(self, population: "Population", function: "Function") -> None:
        pass

    def on_update_before(self, ctx: "UpdateContext") -> None:
        pass

    def on_update_after(self, ctx: "UpdateContext") -> None:
        pass


class CallbackVessel:
    """Aggregates multiple callbacks and dispatches events to all of them."""

    def __init__(self, callbacks: List[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List[Callback]) -> None:
        if not isinstance(callbacks, list):
            raise e.TypeError("`callbacks` should be a list")
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

    def on_evaluate_before(self, population: "Population", function: "Function") -> None:
        for cb in self.callbacks:
            cb.on_evaluate_before(population, function)

    def on_evaluate_after(self, population: "Population", function: "Function") -> None:
        for cb in self.callbacks:
            cb.on_evaluate_after(population, function)

    def on_update_before(self, ctx: "UpdateContext") -> None:
        for cb in self.callbacks:
            cb.on_update_before(ctx)

    def on_update_after(self, ctx: "UpdateContext") -> None:
        for cb in self.callbacks:
            cb.on_update_after(ctx)


class CheckpointCallback(Callback):
    """Periodically saves the optimization model to disk."""

    def __init__(self, file_path: str = "checkpoint.pkl", frequency: int = 0) -> None:
        super().__init__()
        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model) -> None:
        if self.frequency > 0 and iteration % self.frequency == 0:
            opt_model.save(f"iter_{iteration}_{self.file_path}")


class DiscreteSearchCallback(Callback):
    """Maps continuous positions to the nearest allowed discrete values before evaluation."""

    def __init__(self, allowed_values: List[List[Union[int, float]]] | None = None) -> None:
        super().__init__()
        self.allowed_values = allowed_values or []

    def on_task_begin(self, opt_model) -> None:
        n_variables = opt_model.space.population.n_variables
        assert len(self.allowed_values) == n_variables, (
            f"`allowed_values` should have length {n_variables}."
        )

    def on_evaluate_before(self, population: "Population", function: "Function") -> None:
        for i, allowed in enumerate(self.allowed_values):
            allowed_t = torch.tensor(allowed, device=population.device, dtype=torch.float32)
            # positions[:, i, :] has shape (n_agents, n_dims)
            agent_vals = population.positions[:, i, :]  # (n_agents, n_dims)
            diffs = torch.abs(agent_vals.unsqueeze(-1) - allowed_t)  # (n_agents, n_dims, n_allowed)
            nearest_idx = diffs.argmin(dim=-1)  # (n_agents, n_dims)
            population.positions[:, i, :] = allowed_t[nearest_idx]
