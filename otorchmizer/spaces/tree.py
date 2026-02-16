"""Tree-based search space for Genetic Programming."""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.node import Node
from otorchmizer.core.population import Population
from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class TreeSpace(Space):
    """Search space managing both agent positions and expression trees.

    Each agent has an associated tree (Node) used in Genetic Programming.
    Trees are generated using the GROW algorithm.
    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: Union[float, List, Tuple, torch.Tensor],
        upper_bound: Union[float, List, Tuple, torch.Tensor],
        n_terminals: int = 1,
        min_depth: int = 1,
        max_depth: int = 3,
        functions: Optional[List[str]] = None,
        mapping: Optional[List[str]] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
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
            raise e.ValueError("`n_terminals` should be > 0")
        if min_depth <= 0:
            raise e.ValueError("`min_depth` should be > 0")
        if max_depth < min_depth:
            raise e.ValueError("`max_depth` should be >= `min_depth`")

        self.n_terminals = n_terminals
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.functions = functions or []

        self._create_terminals()
        self._create_trees()
        self.build()

        logger.info("Class created.")

    def _create_terminals(self) -> None:
        """Creates terminal value tensors."""

        lb = self.population.lb.squeeze(-1)
        ub = self.population.ub.squeeze(-1)
        n_vars = self.population.n_variables

        self.terminals = []
        for _ in range(self.n_terminals):
            val = r.generate_uniform_random_number(
                low=0.0, high=1.0,
                size=(n_vars, 1),
                device=self.device,
            ) * (ub.unsqueeze(-1) - lb.unsqueeze(-1)) + lb.unsqueeze(-1)
            self.terminals.append(val)

    def _create_trees(self) -> None:
        """Creates a list of trees using the GROW algorithm."""

        self.trees = [
            self.grow(self.min_depth, self.max_depth)
            for _ in range(self.population.n_agents)
        ]
        self.best_tree = copy.deepcopy(self.trees[0])

    def _initialize(self) -> None:
        self.population.initialize_uniform()

    def grow(self, min_depth: int = 1, max_depth: int = 3) -> Node:
        """Creates a random tree based on the GROW algorithm.

        Args:
            min_depth: Minimum depth.
            max_depth: Maximum depth.

        Returns:
            Random expression tree.
        """

        if min_depth == max_depth:
            tid = r.generate_integer_random_number(0, self.n_terminals)
            return Node(tid, "TERMINAL", self.terminals[tid].clone())

        node_id = r.generate_integer_random_number(
            0, len(self.functions) + self.n_terminals
        )

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
