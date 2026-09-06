# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Geometric Semantic Genetic Programming.

References:
    A. Moraglio, K. Krawiec, and C. G. Johnson. Geometric Semantic Genetic Programming.
    Parallel Problem Solving from Nature (2012).
    G. H. de Rosa, J. P. Papa, and L. P. Papa. Feature Selection Using Geometric Semantic
    Genetic Programming. GECCO Companion (2017).

"""

from __future__ import annotations

from typing import Any

import torch

import otorchmizer.utils.exception as e
from otorchmizer.core.node import Node
from otorchmizer.optimizers.evolutionary.gp import GP
from otorchmizer.spaces.tree import TreeSpace
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


def _function(name: str, left: Node, right: Node | None = None) -> Node:
    root = Node(name, "FUNCTION")
    root.left = left
    left.parent = root
    left.flag = True

    if right is not None:
        root.right = right
        right.parent = root
        right.flag = False

    return root


class GSGP(GP):
    """Evolve trees with geometric semantic crossover and mutation."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the optimizer.

        Args:
            params: Parameter overrides applied after the algorithm defaults.

        Notes:
            mutation_step scales the semantic displacement ``R1 - R2`` and defaults to 0.1.

        """

        logger.info("Overriding class: GP -> GSGP.")

        self.mutation_step = 0.1

        super().__init__(params)

        logger.info("Class overrided.")

    @property
    def mutation_step(self) -> float:
        """Return the geometric semantic mutation step."""

        return self._mutation_step

    @mutation_step.setter
    def mutation_step(self, mutation_step: float) -> None:
        if isinstance(mutation_step, bool) or not isinstance(mutation_step, (float, int)):
            raise e.TypeError("`mutation_step` should be a float or integer.")
        if mutation_step < 0:
            raise e.ValueError("`mutation_step` should be greater than or equal to 0.")
        self._mutation_step = float(mutation_step)

    @property
    def prunning_ratio(self) -> float:
        """Return the required zero pruning ratio for whole-tree semantic operators."""

        return self._prunning_ratio

    @prunning_ratio.setter
    def prunning_ratio(self, prunning_ratio: float) -> None:
        if isinstance(prunning_ratio, bool) or not isinstance(prunning_ratio, (float, int)):
            raise e.TypeError("`prunning_ratio` should be a float or integer.")
        if prunning_ratio != 0:
            raise e.ValueError(
                "`prunning_ratio` should be 0 for GSGP because geometric semantic operators use complete trees."
            )
        self._prunning_ratio = 0.0

    @staticmethod
    def _random_terminal(reference: torch.Tensor, name: str) -> Node:
        return Node(name, "TERMINAL", torch.rand_like(reference))

    def _mutate(self, space: TreeSpace, tree: Node, max_nodes: int) -> Node:
        """Apply ``T + mutation_step * (R1 - R2)`` to a parent tree.

        Args:
            space: TreeSpace owning the parent.
            tree: Parent expression tree.
            max_nodes: Unused GP-compatible node limit.

        Returns:
            Independent semantic-mutation tree.

        """

        reference = space.evaluate_tree(tree)
        random_one = self._random_terminal(reference, "R1")
        random_two = self._random_terminal(reference, "R2")
        difference = _function("SUB", random_one, random_two)

        step = Node("MS", "TERMINAL", torch.full_like(reference, self.mutation_step))
        displacement = _function("MUL", step, difference)
        return _function("SUM", tree.clone(), displacement)

    @staticmethod
    def _semantic_child(first: Node, second: Node, mask: torch.Tensor) -> Node:
        left_mask = Node("R", "TERMINAL", mask.clone())
        right_mask = Node("R", "TERMINAL", mask.clone())
        one = Node("ONE", "TERMINAL", torch.ones_like(mask))

        weighted_first = _function("MUL", left_mask, first.clone())
        complement = _function("SUB", one, right_mask)
        weighted_second = _function("MUL", complement, second.clone())
        return _function("SUM", weighted_first, weighted_second)

    def _cross(
        self,
        father: Node,
        mother: Node,
        max_father: int,
        max_mother: int,
    ) -> tuple[Node, Node]:
        """Apply ``R * T1 + (1 - R) * T2`` to two parent trees.

        Args:
            father: First parent tree.
            mother: Second parent tree.
            max_father: Unused GP-compatible node limit for the first parent.
            max_mother: Unused GP-compatible node limit for the second parent.

        Returns:
            Two independent geometric semantic offspring.

        Raises:
            SizeError: Parent semantics have different shapes.
            ValueError: Parent semantics use different devices or dtypes.

        """

        father_position = father.position
        mother_position = mother.position
        if father_position is None or mother_position is None or father_position.shape != mother_position.shape:
            raise e.SizeError("`father.position` and `mother.position` should have matching shapes.")
        if father_position.device != mother_position.device:
            raise e.ValueError("`father.position` and `mother.position` should use the same device.")
        if father_position.dtype != mother_position.dtype:
            raise e.ValueError("`father.position` and `mother.position` should use the same dtype.")

        mask = torch.rand_like(father_position)
        first = self._semantic_child(father, mother, mask)
        second = self._semantic_child(mother, father, mask)
        return first, second
