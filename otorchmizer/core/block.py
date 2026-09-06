# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Block and Cell classes for graph-based (DAG) optimization."""

from __future__ import annotations

import copy
from collections.abc import Callable

import networkx as nx
from networkx import DiGraph

import otorchmizer.utils.exception as e


class Block:
    """Foundation class for graph-based optimization.

    Notes:
        A block wraps a callable and declares its input and output arity.

    """

    def __init__(self, type: str, pointer: Callable, n_input: int, n_output: int) -> None:
        """Initialize a callable block with declared connection sizes.

        Args:
            type: Block role, either "input", "inner", or "output".
            pointer: Callable invoked with the block's positional inputs.
            n_input: Positive number of declared inputs.
            n_output: Positive number of declared outputs.

        Raises:
            TypeError: The pointer is not callable.
            ValueError: The role is unsupported or an arity is not a positive integer.

        """

        self.type = type
        self.pointer = pointer
        self.n_input = n_input
        self.n_output = n_output

    def __call__(self, *args):
        """Invoke the wrapped callable without enforcing its declared arity.

        Args:
            *args: Positional inputs forwarded to the callable.

        Returns:
            The callable's result without conversion.

        """

        return self.pointer(*args)

    @property
    def type(self) -> str:
        """Role used to identify input, intermediate, and output blocks."""

        return self._type

    @type.setter
    def type(self, type: str) -> None:
        if type not in ("input", "inner", "output"):
            raise e.ValueError("`type` should be 'input', 'inner' or 'output'.")
        self._type = type

    @property
    def pointer(self) -> Callable:
        """Callable invoked when the block is evaluated."""

        return self._pointer

    @pointer.setter
    def pointer(self, pointer: Callable) -> None:
        if not callable(pointer):
            raise e.TypeError("`pointer` should be a callable.")
        self._pointer = pointer

    @property
    def n_input(self) -> int:
        """Declared input arity used to validate graph connections."""

        return self._n_input

    @n_input.setter
    def n_input(self, n_input: int) -> None:
        if not isinstance(n_input, int) or n_input <= 0:
            raise e.ValueError("`n_input` should be a positive integer.")
        self._n_input = n_input

    @property
    def n_output(self) -> int:
        """Declared output arity used to validate graph connections."""

        return self._n_output

    @n_output.setter
    def n_output(self, n_output: int) -> None:
        if not isinstance(n_output, int) or n_output <= 0:
            raise e.ValueError("`n_output` should be a positive integer.")
        self._n_output = n_output


class InputBlock(Block):
    """Entry-point block (identity function)."""

    def __init__(self, n_input: int, n_output: int) -> None:
        """Initialize an input block that returns its arguments as a tuple.

        Args:
            n_input: Positive number of declared inputs.
            n_output: Positive number of declared outputs.

        """

        super().__init__("input", lambda *args: args, n_input, n_output)


class InnerBlock(Block):
    """Block for intermediate computation."""

    def __init__(self, pointer: Callable, n_input: int, n_output: int) -> None:
        """Initialize an intermediate block around a callable.

        Args:
            pointer: Callable invoked with the block's positional inputs.
            n_input: Positive number of declared inputs.
            n_output: Positive number of declared outputs.

        """

        super().__init__("inner", pointer, n_input, n_output)


class OutputBlock(Block):
    """Exit-point block (identity function)."""

    def __init__(self, n_input: int, n_output: int) -> None:
        """Initialize an output block that returns its arguments as a tuple.

        Args:
            n_input: Positive number of declared inputs.
            n_output: Positive number of declared outputs.

        """

        super().__init__("output", lambda *args: args, n_input, n_output)


class Cell(DiGraph):
    """A directed graph of blocks with arity-matched connections.

    Notes:
        Evaluation follows all simple paths from the first input block to the first output block.
        Graphs with cycles or missing endpoints return no outputs.

    """

    def __init__(self, blocks: list[Block], edges: list[tuple[int, int]]) -> None:
        """Build a graph from indexed blocks and compatible edges.

        Args:
            blocks: Blocks assigned integer node indices in list order.
            edges: Source and target indices for proposed connections.

        Notes:
            Edges with missing nodes or mismatched arities are silently omitted.
            Construction does not reject cycles or require input and output blocks.

        """

        super().__init__()

        for i, block in enumerate(blocks):
            self.add_node(i, block=block)

        for u, v in edges:
            if (
                u in self.nodes
                and v in self.nodes
                and self.nodes[u]["block"].n_output == self.nodes[v]["block"].n_input
            ):
                self.add_edge(u, v)

    def __call__(self, *args) -> list:
        """Evaluate each input-to-output path independently.

        Args:
            *args: Inputs deep-copied separately for each path.

        Returns:
            One output tuple per path, or an empty list when the graph is invalid or has no paths.

        """

        if not self.valid:
            return []

        paths = list(nx.all_simple_paths(self, self.input_idx, self.output_idx))
        outputs = []

        for path in paths:
            current_args = copy.deepcopy(args)
            for node in path:
                current_args = self.nodes[node]["block"](*current_args)
                if not isinstance(current_args, tuple):
                    current_args = (current_args,)
            outputs.append(current_args)

        return outputs

    @property
    def input_idx(self) -> int:
        """Index of the first input block, or -1 if none exists."""

        for node in self.nodes:
            if self.nodes[node]["block"].type == "input":
                return node
        return -1

    @property
    def output_idx(self) -> int:
        """Index of the first output block, or -1 if none exists."""

        for node in self.nodes:
            if self.nodes[node]["block"].type == "output":
                return node
        return -1

    @property
    def valid(self) -> bool:
        """Whether input and output blocks exist and the graph is acyclic."""

        if self.input_idx == -1 or self.output_idx == -1:
            return False
        return nx.is_directed_acyclic_graph(self)
