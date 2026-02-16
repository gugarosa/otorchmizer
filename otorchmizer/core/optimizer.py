"""Optimizer base class and UpdateContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.core.population import Population
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class UpdateContext:
    """All information an optimizer might need during update().

    Every optimizer receives the same context — use what you need, ignore the rest.
    Eliminates the fragile inspect.signature() dynamic wiring from Opytimizer.
    """

    space: Any  # Space instance (forward ref to avoid circular import)
    function: Function
    iteration: int
    n_iterations: int
    device: torch.device


class Optimizer:
    """Base class for all optimization algorithms.

    Subclasses MUST implement update(ctx: UpdateContext).
    Subclasses MAY override evaluate() and compile().
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.algorithm = self.__class__.__name__
        self.params = {}
        self.built = False
        self._compiled_update = None

        self.build(params)

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: str) -> None:
        if not isinstance(algorithm, str):
            raise e.TypeError("`algorithm` should be a string")
        self._algorithm = algorithm

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        self._built = built

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @params.setter
    def params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise e.TypeError("`params` should be a dictionary")
        self._params = params

    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
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

        Called once before the optimization loop starts. Subclasses
        should create velocities, memories, etc. on `population.device`.

        Args:
            population: Population instance with device and shape info.
        """

        pass

    def torch_compile(self, **kwargs) -> None:
        """JIT-compiles the update method via ``torch.compile``.

        After calling this, subsequent ``update()`` invocations go through
        the compiled graph, fusing operations and eliminating Python overhead
        for an additional 2–5× speedup on compatible hardware.

        Args:
            **kwargs: Forwarded to ``torch.compile`` (e.g., ``mode="reduce-overhead"``
                      or ``fullgraph=True``).

        Example::

            opt = PSO()
            opt.compile(pop)
            opt.torch_compile(mode="reduce-overhead")
            for i in range(n):
                opt.update(ctx)  # runs through compiled graph
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

        Default: vectorized evaluation with no per-agent loop.
        Override when custom evaluation logic is needed (e.g., PSO local bests).

        Args:
            population: Population to evaluate.
            function: Objective function.
        """

        population.fitness = function(population.positions)
        population.update_best()

    def update(self, ctx: UpdateContext) -> None:
        """Applies the algorithm's position-update rule.

        MUST be implemented by every optimizer subclass.
        Should use ONLY tensor operations — no Python loops over agents.

        Args:
            ctx: UpdateContext with all available optimization state.
        """

        raise NotImplementedError(
            f"{self.algorithm} must implement update(ctx: UpdateContext)"
        )

    def __call__(self, ctx: UpdateContext) -> None:
        """Dispatches to compiled or regular update."""

        if self._compiled_update is not None:
            self._compiled_update(ctx)
        else:
            self.update(ctx)

    def __repr__(self) -> str:
        compiled = ", compiled=True" if self._compiled_update is not None else ""
        return f"{self.algorithm}(params={self.params}, built={self.built}{compiled})"
