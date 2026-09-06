# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Constrained single-objective function wrapper."""

from __future__ import annotations

from collections.abc import Callable

import torch

from otorchmizer.core.function import Function, _reject_nan


class ConstrainedFunction(Function):
    """Wraps an objective function with constraint penalties.

    Notes:
        Constraints return True when satisfied and False otherwise.
        Each violation applies fitness += penalty * abs(fitness) to the current fitness.
        The objective and constraints receive individual agent positions unless batch=True.
        Constraint wrappers share the objective's vectorization and fallback behavior.
        Changes to the mutable constraint list are reflected on the next evaluation.

    """

    def __init__(
        self,
        pointer: Callable,
        constraints: list[Callable] | None,
        penalty: float = 0.0,
        batch: bool = False,
    ) -> None:
        """Wrap an objective and configure sequential constraint penalties.

        Args:
            pointer: Callable returning a fitness value.
            constraints: List of constraint callables, or None for no constraints.
            penalty: Penalty factor for violated constraints.
            batch: If True, all callables handle full population tensors.

        Raises:
            TypeError: The objective is not callable, constraints are not a list, or the penalty is not numeric.
            ValueError: The penalty is negative.

        """

        super().__init__(pointer, batch)

        self.constraints = [] if constraints is None else constraints
        self.penalty = penalty

        if not isinstance(self.constraints, list):
            raise TypeError("`constraints` should be a list.")
        if not all(callable(constraint) for constraint in self.constraints):
            raise TypeError("`constraints` must contain only callables.")
        if not isinstance(self.penalty, (float, int)):
            raise TypeError("`penalty` should be a float or integer.")
        if self.penalty < 0:
            raise ValueError("`penalty` should be >= 0.")

        self._constraint_functions = [Function(constraint, batch=self.batch) for constraint in self.constraints]

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates fitness with constraint penalties.

        Args:
            positions: Population tensor shaped (n_agents, n_variables, n_dimensions).

        Returns:
            Penalized fitness tensor of shape (n_agents,).

        Raises:
            ValueError: Eager objective or penalized fitness contains NaN.
            RuntimeError: Compiled objective or penalized fitness contains NaN.

        """

        fitness = super().__call__(positions)

        if len(self._constraint_functions) != len(self.constraints) or any(
            wrapped._raw_pointer is not raw or wrapped.batch != self.batch
            for raw, wrapped in zip(self.constraints, self._constraint_functions)
        ):
            self._constraint_functions = [Function(constraint, batch=self.batch) for constraint in self.constraints]

        for constraint in self._constraint_functions:
            satisfied = constraint(positions)
            mask = ~satisfied.bool()
            fitness = torch.where(mask & (self.penalty != 0), fitness + self.penalty * fitness.abs(), fitness)
        _reject_nan(fitness)

        return fitness
