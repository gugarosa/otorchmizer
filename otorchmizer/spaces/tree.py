# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tree-based search space for Genetic Programming."""

from __future__ import annotations

import torch

import otorchmizer.math.random as r
import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e
from otorchmizer.core.device import DeviceManager
from otorchmizer.core.node import Node
from otorchmizer.core.space import Space
from otorchmizer.utils import logging

logger = logging.get_logger(__name__)


class TreeSpace(Space):
    """Search space combining expression-tree genotypes with tensor phenotypes."""

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
            TypeError: A terminal count, depth, or function collection has an invalid type.
            ValueError: A terminal count, depth, or function name is invalid.

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
        self.device = self.population.positions.device
        self.population.device = self.device

        if not isinstance(n_terminals, int):
            raise e.TypeError("`n_terminals` should be an integer.")
        if n_terminals <= 0:
            raise e.ValueError("`n_terminals` should be greater than 0.")
        if not isinstance(min_depth, int):
            raise e.TypeError("`min_depth` should be an integer.")
        if min_depth <= 0:
            raise e.ValueError("`min_depth` should be greater than 0.")
        if not isinstance(max_depth, int):
            raise e.TypeError("`max_depth` should be an integer.")
        if max_depth < min_depth:
            raise e.ValueError("`max_depth` should be greater than or equal to `min_depth`.")
        if functions is not None and not isinstance(functions, list):
            raise e.TypeError("`functions` should be a list.")

        functions = functions or []
        unsupported = [name for name in functions if name not in c.FUNCTION_N_ARGS]
        if unsupported:
            raise e.ValueError(f"`functions` contains unsupported functions: {unsupported}.")

        self.n_terminals = n_terminals
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.functions = functions

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
            val = torch.rand(
                n_vars,
                1,
                device=self.device,
                dtype=self.population.dtype,
            ) * (ub.unsqueeze(-1) - lb.unsqueeze(-1)) + lb.unsqueeze(-1)
            self.terminals.append(val)

    def _create_trees(self) -> None:
        self.trees = [self.grow(self.min_depth, self.max_depth) for _ in range(self.population.n_agents)]
        self.best_tree = self.trees[0].clone()
        self._best_tree_needs_evaluation = False

    def _initialize(self) -> None:
        self.sync_positions()
        self.population.best_position = self.population.positions[0].clone()

    def evaluate_tree(self, tree: Node) -> torch.Tensor:
        """Evaluates and bounds one expression tree as a population position.

        Args:
            tree: Root node of the expression tree.

        Returns:
            Position tensor shaped ``(n_variables, 1)`` on the space device.

        Raises:
            TypeError: The tree is not a Node.
            SizeError: The expression result has an incompatible shape.
            ValueError: The expression result has an incompatible device, dtype, or non-finite value.

        Notes:
            Finite values are clipped to the variable bounds. Invalid expressions are rejected rather than mapped
            onto an ordinary candidate that could receive a favorable objective value.

        """

        if not isinstance(tree, Node):
            raise e.TypeError("`tree` should be a Node.")

        position = tree.position
        expected = (self.population.n_variables, self.population.n_dimensions)
        if position is None or tuple(position.shape) != expected:
            actual = None if position is None else tuple(position.shape)
            raise e.SizeError(f"`tree.position` must have shape {expected}, but got {actual}.")
        if position.device != self.device:
            raise e.ValueError(f"`tree.position.device` should be {self.device}, but got {position.device}.")
        if position.dtype != self.population.dtype:
            raise e.ValueError(f"`tree.position.dtype` should be {self.population.dtype}, but got {position.dtype}.")
        if not torch.isfinite(position).all():
            raise e.ValueError("`tree.position` should contain only finite values.")

        return position.clamp(min=self.population.lb, max=self.population.ub)

    def sync_positions(self) -> None:
        """Synchronize tensor positions with trees and invalidate their fitness.

        Raises:
            SizeError: The number of trees differs from the number of agents.

        """

        if len(self.trees) != self.population.n_agents:
            raise e.SizeError(f"`trees` must contain {self.population.n_agents} roots, but got {len(self.trees)}.")

        self.population.positions = torch.stack([self.evaluate_tree(tree) for tree in self.trees])
        self.population.fitness.fill_(torch.inf)
        if not torch.isfinite(self.population.best_fitness) and not self._best_tree_needs_evaluation:
            self.best_tree = self.trees[0].clone()
            self.population.best_position = self.population.positions[0].clone()

    def validate_positions(self) -> None:
        """Reject population or archived-best positions changed independently from their trees.

        Raises:
            ValueError: Population positions do not equal the bounded tree phenotypes.

        Notes:
            Tree optimizers support callbacks that observe positions, but not callbacks that transform them.
            The canonical tree-derived positions are restored before raising.

        """

        expected = torch.stack([self.evaluate_tree(tree) for tree in self.trees])
        expected_best = self.evaluate_tree(self.best_tree)
        positions = self.population.positions
        matches = (
            positions.shape == expected.shape
            and positions.device == expected.device
            and positions.dtype == expected.dtype
            and torch.equal(positions, expected)
            and self.population.best_position.shape == expected_best.shape
            and self.population.best_position.device == expected_best.device
            and self.population.best_position.dtype == expected_best.dtype
            and torch.equal(self.population.best_position, expected_best)
        )
        if not matches:
            self.population.positions = expected
            self.population.best_position = expected_best
            raise e.ValueError(
                "`population.positions` and `population.best_position` cannot be changed independently of their trees; "
                "tree optimizers support observational callbacks only."
            )

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> TreeSpace:
        """Move the complete tree space to a device and floating-point dtype.

        Args:
            device: Target device or "auto".
            dtype: Target floating-point dtype, or None to preserve the current dtype.

        Returns:
            This TreeSpace after an in-place transfer.

        Raises:
            TypeError: The dtype is not a torch dtype.
            ValueError: The dtype is not floating point or a transferred tree is invalid.

        Notes:
            This moves terminal prototypes, every current tree, the archived best tree, and all population tensors.
            Current and best fitness are invalidated. A previously evaluated best tree is retained and re-evaluated
            by GP or GSGP on the next evaluation, even when it is absent from the current population.
            Prefer transferring before constructing Otorchmizer. After binding, call optimizer.rebind(space) so
            compiled dispatch and optimizer state are rebuilt for the transferred population.

        """

        target_dtype = dtype or self.population.dtype
        if not isinstance(target_dtype, torch.dtype):
            raise e.TypeError("`dtype` should be a torch.dtype.")
        if not target_dtype.is_floating_point:
            raise e.ValueError("`dtype` should be a floating-point dtype.")

        target_device = DeviceManager(device).device
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())

        preserve_best = self._best_tree_needs_evaluation or torch.isfinite(self.population.best_fitness).item()
        self._best_tree_needs_evaluation = preserve_best
        self.terminals = [terminal.to(device=target_device, dtype=target_dtype) for terminal in self.terminals]

        for tree in [*self.trees, self.best_tree]:
            for node in tree.pre_order:
                if node.value is not None:
                    node.value = node.value.to(device=target_device, dtype=target_dtype)

        self.population.to(target_device, dtype=target_dtype)
        self.device = target_device
        self.sync_positions()
        self.population.best_position = self.evaluate_tree(self.best_tree)
        self.population.best_fitness.fill_(torch.inf)
        return self

    def grow(self, min_depth: int = 1, max_depth: int = 3) -> Node:
        """Create a random tree with the GROW algorithm.

        Args:
            min_depth: Minimum depth.
            max_depth: Maximum depth.

        Returns:
            Random expression tree.

        Raises:
            TypeError: A depth is not an integer.
            ValueError: A depth is invalid.

        Notes:
            Function nodes are forced until min_depth and terminal nodes are forced at max_depth.
            When no functions are configured, a terminal tree is returned.

        """

        if not isinstance(min_depth, int):
            raise e.TypeError("`min_depth` should be an integer.")
        if not isinstance(max_depth, int):
            raise e.TypeError("`max_depth` should be an integer.")
        if min_depth < 0:
            raise e.ValueError("`min_depth` should be greater than or equal to 0.")
        if max_depth < min_depth:
            raise e.ValueError("`max_depth` should be greater than or equal to `min_depth`.")

        return self._grow(0, min_depth, max_depth)

    def _grow(self, depth: int, min_depth: int, max_depth: int) -> Node:
        if depth >= max_depth or not self.functions:
            tid = r.generate_integer_random_number(0, self.n_terminals)
            return Node(tid, "TERMINAL", self.terminals[tid].clone())

        if depth < min_depth:
            node_id = r.generate_integer_random_number(0, len(self.functions))
        else:
            node_id = r.generate_integer_random_number(0, len(self.functions) + self.n_terminals)

        if node_id >= len(self.functions):
            tid = node_id - len(self.functions)
            return Node(tid, "TERMINAL", self.terminals[tid].clone())

        fn_name = self.functions[node_id]
        fn_node = Node(fn_name, "FUNCTION")

        for i in range(c.FUNCTION_N_ARGS[fn_name]):
            child = self._grow(depth + 1, min_depth, max_depth)

            if not i:
                fn_node.left = child
            else:
                fn_node.right = child
                child.flag = False

            child.parent = fn_node

        return fn_node
