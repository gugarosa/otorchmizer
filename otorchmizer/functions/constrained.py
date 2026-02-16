"""Constrained single-objective function wrapper."""

from __future__ import annotations

from typing import List

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.function import Function
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class ConstrainedFunction(Function):
    """Wraps an objective function with constraint penalties.

    Constraints are callable predicates: return True if satisfied, False otherwise.
    When a constraint is violated, the fitness is penalized by
    `fitness += penalty * |fitness|`.

    Both the pointer and constraints should handle individual agent positions
    unless batch=True.
    """

    def __init__(
        self,
        pointer: callable,
        constraints: List[callable],
        penalty: float = 0.0,
        batch: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            pointer: Callable returning a fitness value.
            constraints: List of constraint callables.
            penalty: Penalty factor for violated constraints.
            batch: If True, all callables handle full population tensors.
        """

        logger.info("Creating class: ConstrainedFunction.")

        super().__init__(pointer, batch)

        self.constraints = constraints or []
        self.penalty = penalty

        if not isinstance(self.constraints, list):
            raise e.TypeError("`constraints` should be a list")
        if not isinstance(self.penalty, (float, int)):
            raise e.TypeError("`penalty` should be a float or integer")
        if self.penalty < 0:
            raise e.ValueError("`penalty` should be >= 0")

        logger.debug("Constraints: %d | Penalty: %s.", len(self.constraints), self.penalty)
        logger.info("Class created.")

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluates fitness with constraint penalties.

        Args:
            positions: (n_agents, n_variables, n_dimensions).

        Returns:
            Penalized fitness tensor of shape (n_agents,).
        """

        fitness = self._fn(positions)

        for constraint in self.constraints:
            if self.batch:
                # constraint returns bool tensor (n_agents,)
                satisfied = constraint(positions)
            else:
                # Auto-vectorize constraint
                try:
                    satisfied = torch.vmap(constraint)(positions)
                except Exception:
                    satisfied = torch.stack([constraint(positions[i])
                                            for i in range(positions.shape[0])])

            mask = ~satisfied.bool()
            fitness = fitness + mask.float() * self.penalty * fitness.abs()

        return fitness
