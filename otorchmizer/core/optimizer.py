# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Optimizer base class and UpdateContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

if TYPE_CHECKING:
    from otorchmizer.core.space import Space

logger = logging.get_logger(__name__)


@dataclass
class UpdateContext:
    """Shared optimization state passed to each update.

    Attributes:
        space (otorchmizer.core.space.Space): Search space containing the population and its bounds.
        function (otorchmizer.core.function.Function): Objective used to evaluate candidates.
        iteration: Zero-based iteration index within the current run.
        n_iterations: Requested iteration count for the current run.
        device: Device used by the search space.

    Notes:
        Every optimizer receives the same context and uses only the fields it needs.
        Explicit fields replace signature inspection and dynamic argument wiring.

    """

    space: Space
    function: Function
    iteration: int
    n_iterations: int
    device: torch.device


class Optimizer:
    """Base class for all optimization algorithms.

    Notes:
        Subclasses must implement update(ctx) and may override evaluate() and compile().

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize algorithm metadata and apply parameter overrides.

        Args:
            params: Attribute overrides applied by build.

        """

        self.algorithm = self.__class__.__name__
        self.params = {}
        self.built = False
        self._compiled_update = None

        self.build(params)

    @property
    def algorithm(self) -> str:
        """Algorithm name initialized from the concrete class name."""

        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: str) -> None:
        if not isinstance(algorithm, str):
            raise e.TypeError("`algorithm` should be a string.")
        self._algorithm = algorithm

    @property
    def built(self) -> bool:
        """Whether optimizer parameter initialization has completed."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        self._built = built

    @property
    def params(self) -> dict[str, Any]:
        """Mutable dictionary of custom parameter overrides."""

        return self._params

    @params.setter
    def params(self, params: dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise e.TypeError("`params` should be a dictionary.")
        self._params = params

    def build(self, params: dict[str, Any] | None = None) -> None:
        """Builds the optimizer by applying parameter overrides.

        Args:
            params: Key-value parameters to override defaults.

        """

        if params:
            self.params.update(params)
            for k, v in params.items():
                setattr(self, k, v)

        self.built = True

        logger.debug(
            "Algorithm: %s | Custom Parameters: %s | Built: %s.",
            self.algorithm,
            str(params),
            self.built,
        )

    def compile(self, population: Population) -> None:
        """Pre-allocates algorithm-specific state tensors.

        Args:
            population: Population instance with device and shape info.

        Notes:
            Called once when the optimization engine is constructed.
            Subclasses should allocate velocities, memories, and other state on the population's device.
            The base implementation does nothing.

        """

        pass

    def torch_compile(self, **kwargs) -> None:
        """JIT-compiles the update method via ``torch.compile``.

        Args:
            **kwargs: Keyword arguments forwarded to torch.compile.

        Notes:
            Subsequent optimizer(ctx) calls dispatch through the compiled function.
            The engine uses this entry point, while direct update(ctx) calls remain eager.
            The compile mode is "reduce-overhead" unless explicitly overridden.
            Performance depends on the algorithm, backend, workload, and hardware.

        Examples:
            Compile optimizer dispatch after allocating algorithm state::

                opt = PSO()
                opt.compile(pop)
                opt.torch_compile(mode="reduce-overhead")
                for i in range(n):
                    opt(ctx)

        """

        kwargs.setdefault("mode", "reduce-overhead")
        self._compiled_update = torch.compile(self.update, **kwargs)
        logger.info(
            "torch.compile enabled for %s (mode=%s)",
            self.algorithm,
            kwargs.get("mode"),
        )

    def evaluate(self, population: Population, function: Function) -> None:
        """Batch-evaluates all agents and updates global best.

        Args:
            population: Population to evaluate.
            function: Objective function.

        Notes:
            Evaluation delegates to the objective's batching strategy.
            Override this method for custom tracking such as PSO's per-agent best positions.

        """

        population.fitness = function(population.positions)
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Applies the algorithm's position-update rule.

        Args:
            ctx: UpdateContext with all available optimization state.

        Raises:
            NotImplementedError: The subclass has not implemented its update rule.

        Notes:
            Every optimizer subclass must implement this method.
            Prefer tensor operations over Python loops across agents.

        """

        raise NotImplementedError(f"`{self.algorithm}` must implement update(ctx: UpdateContext).")

    def __call__(self, ctx: UpdateContext) -> None:
        """Dispatch to the compiled update when available, otherwise the eager update.

        Args:
            ctx: Current optimization state passed unchanged to the update.

        """

        if self._compiled_update is not None:
            self._compiled_update(ctx)
        else:
            self.update(ctx)

    def __repr__(self) -> str:
        compiled = ", compiled=True" if self._compiled_update is not None else ""
        return f"{self.algorithm}(params={self.params}, built={self.built}{compiled})"
