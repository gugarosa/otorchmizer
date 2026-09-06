# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tree-based search space for Genetic Programming."""

from __future__ import annotations

import copy

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.node import Node
from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class TreeSpace(Space):
    """Search space combining agent positions with expression trees."""

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        upper_bound: float | list[float] | tuple[float, ...] | torch.Tensor,
        n_terminals: int = 1,
        min_depth: int = 1,
        max_depth: int = 3,
        functions: list[str] | None = None,
        mapping: list[str] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        """Initialize a tree-based search space.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            lower_bound: Lower bound for each decision variable.
            upper_bound: Upper bound for each decision variable.
            n_terminals: Number of terminal value tensors.
            min_depth: Minimum tree depth.
            max_depth: Maximum tree depth.
            functions: Function node names available to generated trees.
            mapping: Human-readable names for the decision variables.
            device: Device used to store population tensors.

        Raises:
            ValueError: If terminal count or depth constraints are invalid.

        Notes:
            Each agent has an associated expression tree generated with the GROW algorithm.

        """

        logger.info("Creating class: TreeSpace.")

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            device=device,
        )

        if n_terminals <= 0:
            raise e.ValueError("`n_terminals` should be greater than 0.")
        if min_depth <= 0:
            raise e.ValueError("`min_depth` should be greater than 0.")
        if max_depth < min_depth:
            raise e.ValueError("`max_depth` should be greater than or equal to `min_depth`.")

        self.n_terminals = n_terminals
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.functions = functions or []

        self._create_terminals()
        self._create_trees()
        self.build()

        logger.info("Class created.")

    def _create_terminals(self) -> None:
        lb = self.population.lb.squeeze(-1)
        ub = self.population.ub.squeeze(-1)
        n_vars = self.population.n_variables

        self.terminals = []
        for _ in range(self.n_terminals):
            val = r.generate_uniform_random_number(
                low=0.0,
                high=1.0,
                size=(n_vars, 1),
                device=self.device,
            ) * (ub.unsqueeze(-1) - lb.unsqueeze(-1)) + lb.unsqueeze(-1)
            self.terminals.append(val)

    def _create_trees(self) -> None:
        self.trees = [self.grow(self.min_depth, self.max_depth) for _ in range(self.population.n_agents)]
        self.best_tree = copy.deepcopy(self.trees[0])

    def _initialize(self) -> None:
        self.population.initialize_uniform()

    def grow(self, min_depth: int = 1, max_depth: int = 3) -> Node:
        """Create a random tree with the GROW algorithm.

        Args:
            min_depth: Minimum depth.
            max_depth: Maximum depth.

        Returns:
            Random expression tree.

        """

        if min_depth == max_depth:
            tid = r.generate_integer_random_number(0, self.n_terminals)
            return Node(tid, "TERMINAL", self.terminals[tid].clone())

        node_id = r.generate_integer_random_number(0, len(self.functions) + self.n_terminals)

        if node_id >= len(self.functions):
            tid = node_id - len(self.functions)
            return Node(tid, "TERMINAL", self.terminals[tid].clone())

        fn_name = self.functions[node_id]
        fn_node = Node(fn_name, "FUNCTION")

        for i in range(c.FUNCTION_N_ARGS[fn_name]):
            child = self.grow(min_depth + 1, max_depth)

            if not i:
                fn_node.left = child
            else:
                fn_node.right = child
                child.flag = False

            child.parent = fn_node

        return fn_node
