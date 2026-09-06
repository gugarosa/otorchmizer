# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Coordinate tensor optimization, per-run callbacks, and trusted checkpoints."""

from __future__ import annotations

import operator
import time
from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, SupportsIndex

import dill
from tqdm import tqdm

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.space import Space
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction
from otorchmizer.utils.callback import Callback
from otorchmizer.utils.history import History


def _emit(callbacks: Sequence[Callback] | None, event: str, *args: Any) -> None:
    if callbacks is not None:
        for callback in callbacks:
            getattr(callback, event)(*args)


class Otorchmizer:
    """Coordinate a space, strategy, and scalar objective without cloning their state."""

    def __init__(
        self,
        space: Space,
        optimizer: Optimizer,
        function: Callable,
        save_agents: bool = False,
    ) -> None:
        """Validate components and compile strategy state once.

        Args:
            space: Built space owning the population.
            optimizer: Strategy whose state is bound and compiled during construction.
            function: Scalar callable or an existing tensor objective adapter.
            save_agents: Whether to retain population position snapshots.

        Raises:
            TypeError: A component or history option has an invalid type.
            RuntimeError: The space or optimizer has not been built.

        """

        if not isinstance(space, Space):
            raise TypeError("`space` must be a Space.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("`optimizer` must be an Optimizer.")
        if not space.built:
            raise RuntimeError("`space` must be built before using Otorchmizer.")
        if not optimizer.built:
            raise RuntimeError("`optimizer` must be built before using Otorchmizer.")

        history = History(save_agents=save_agents)
        self.function = function
        self.space = space
        self.optimizer = optimizer
        self.history = history
        self.iteration = 0
        self.total_iterations = 0
        self.n_iterations = 0

        self.optimizer.bind(space)
        self.optimizer.compile(space.population)

    @property
    def function(self) -> Function | MultiObjectiveFunction:
        """Objective adapter used by subsequent evaluation and update calls.

        Raw scalar callables are wrapped once; existing adapters retain their batching behavior.
        Assignment changes future dispatch, not previously recorded scores, history, or optimizer state.
        Use a fresh space and optimizer for an independent objective rather than mixing incompatible scores.

        """

        return self._function

    @function.setter
    def function(self, function: Callable) -> None:
        self._function = function if isinstance(function, (Function, MultiObjectiveFunction)) else Function(function)

    def _make_context(self) -> UpdateContext:
        return UpdateContext(
            space=self.space,
            function=self.function,
            iteration=self.iteration,
            n_iterations=self.n_iterations,
            device=self.space.device,
        )

    def evaluate(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Evaluate live model state between ordered callback hooks.

        Args:
            callbacks: Per-call callback sequence, or None.

        Notes:
            Replacements made by before-evaluation callbacks are resolved before evaluation.
            Exceptions propagate without an after hook or rollback.

        """

        self.optimizer.validate_space(self.space)
        _emit(callbacks, "on_evaluate_before", self.space.population, self.function)
        self.optimizer.validate_space(self.space)
        self.optimizer.evaluate(self.space.population, self.function)
        self.optimizer.validate_space(self.space)
        _emit(callbacks, "on_evaluate_after", self.space.population, self.function)
        self.optimizer.validate_space(self.space)

    def update(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Update positions between callbacks, then enforce bounds.

        Args:
            callbacks: Per-call callback sequence, or None.

        Notes:
            Context fields are read-only stage snapshots. Referenced state remains live.
            Assign replacements on the model, not the context; the update resolves a fresh context after
            before-update callbacks. After-update hooks run before clipping.
            Tree optimizers retain their observational-callback restrictions.

        """

        self.optimizer.validate_space(self.space)
        _emit(callbacks, "on_update_before", self._make_context())
        self.optimizer.validate_space(self.space)
        self.optimizer(self._make_context())
        self.optimizer.validate_space(self.space)
        _emit(callbacks, "on_update_after", self._make_context())
        self.optimizer.validate_space(self.space)
        self.space.clip()
        self.optimizer.validate_space(self.space)

    def start(
        self,
        n_iterations: SupportsIndex = 1,
        callbacks: Sequence[Callback] | None = None,
        *,
        progress: bool = False,
    ) -> None:
        """Run additional iterations in place without recompiling strategy state.

        Args:
            n_iterations: Nonnegative integer budget.
            callbacks: Ordered callbacks for this invocation only.
            progress: Whether to display a progress bar.

        Raises:
            TypeError: The budget, callback sequence, or progress option has an invalid type.
            ValueError: The budget is negative.

        Notes:
            A zero budget performs task hooks and initial evaluation without updates.
            Iteration-local counters restart; total iterations and history accumulate.
            History precedes iteration-end callbacks. Elapsed time follows normal task completion.
            Exceptions propagate without rollback or a guaranteed task-end hook.

        """

        try:
            iterations = operator.index(n_iterations)
        except TypeError as error:
            raise TypeError("`n_iterations` must be an integer.") from error
        if iterations < 0:
            raise ValueError("`n_iterations` must be non-negative.")
        if callbacks is not None and (
            not isinstance(callbacks, Sequence) or any(not isinstance(callback, Callback) for callback in callbacks)
        ):
            raise TypeError("`callbacks` must be a sequence of Callback instances.")
        if not isinstance(progress, bool):
            raise TypeError("`progress` must be a boolean.")

        self.n_iterations = iterations
        start_time = time.perf_counter()
        _emit(callbacks, "on_task_begin", self)
        self.evaluate(callbacks)

        with tqdm(total=iterations, ascii=True, disable=not progress) as bar:
            for t in range(iterations):
                self.total_iterations += 1
                self.iteration = t
                _emit(callbacks, "on_iteration_begin", self.total_iterations, self)
                self.update(callbacks)
                self.evaluate(callbacks)

                if progress:
                    bar.set_postfix(fitness=self.space.population.best_fitness.item())
                    bar.update()
                self.history.dump(
                    best_agent=(self.space.population.best_position, self.space.population.best_fitness),
                    positions=self.space.population.positions,
                    fitness=self.space.population.fitness,
                )
                _emit(callbacks, "on_iteration_end", self.total_iterations, self)
                self.optimizer.validate_space(self.space)

        _emit(callbacks, "on_task_end", self)
        self.optimizer.validate_space(self.space)
        self.history.dump(time=time.perf_counter() - start_time)

    def save(self, file_path: str | PathLike[str]) -> None:
        """Serialize model state without retaining per-run callbacks or compiled dispatch.

        Args:
            file_path: Output checkpoint path whose parent already exists.

        """

        with open(file_path, "wb") as output:
            dill.dump(self, output)

    @classmethod
    def load(cls, file_path: str | PathLike[str]) -> Otorchmizer:
        """Restore a trusted checkpoint without recompiling optimizer buffers.

        Args:
            file_path: Input checkpoint path.

        Returns:
            Restored model using eager optimizer dispatch.

        Warning:
            Dill deserialization can execute code. Load only trusted checkpoints.

        """

        with open(file_path, "rb") as source:
            return dill.load(source)
