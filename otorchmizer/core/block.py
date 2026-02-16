"""Block and Cell classes for graph-based (DAG) optimization."""

from __future__ import annotations

import copy
from typing import Generator, List, Tuple

import networkx as nx
from networkx import DiGraph

import otorchmizer.utils.exception as e


class Block:
    """Foundation class for graph-based optimization.

    A Block wraps a callable and declares its input/output arity.
    """

    def __init__(self, type: str, pointer: callable,
                 n_input: int, n_output: int) -> None:
        self.type = type
        self.pointer = pointer
        self.n_input = n_input
        self.n_output = n_output

    def __call__(self, *args):
        return self.pointer(*args)

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        if type not in ("input", "inner", "output"):
            raise e.ValueError("`type` should be 'input', 'inner' or 'output'")
        self._type = type

    @property
    def pointer(self) -> callable:
        return self._pointer

    @pointer.setter
    def pointer(self, pointer: callable) -> None:
        if not callable(pointer):
            raise e.TypeError("`pointer` should be a callable")
        self._pointer = pointer

    @property
    def n_input(self) -> int:
        return self._n_input

    @n_input.setter
    def n_input(self, n_input: int) -> None:
        if not isinstance(n_input, int) or n_input <= 0:
            raise e.ValueError("`n_input` should be a positive integer")
        self._n_input = n_input

    @property
    def n_output(self) -> int:
        return self._n_output

    @n_output.setter
    def n_output(self, n_output: int) -> None:
        if not isinstance(n_output, int) or n_output <= 0:
            raise e.ValueError("`n_output` should be a positive integer")
        self._n_output = n_output


class InputBlock(Block):
    """Entry-point block (identity function)."""

    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__("input", lambda *args: args, n_input, n_output)


class InnerBlock(Block):
    """Block for intermediate computation."""

    def __init__(self, pointer: callable, n_input: int, n_output: int) -> None:
        super().__init__("inner", pointer, n_input, n_output)


class OutputBlock(Block):
    """Exit-point block (identity function)."""

    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__("output", lambda *args: args, n_input, n_output)


class Cell(DiGraph):
    """A DAG of Blocks with validated edge connections.

    Forward pass evaluates all simple paths from input to output.
    """

    def __init__(self, blocks: List[Block],
                 edges: List[Tuple[int, int]]) -> None:
        super().__init__()

        for i, block in enumerate(blocks):
            self.add_node(i, block=block)

        for u, v in edges:
            if (u in self.nodes and v in self.nodes
                    and self.nodes[u]["block"].n_output == self.nodes[v]["block"].n_input):
                self.add_edge(u, v)

    def __call__(self, *args) -> list:
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
        for node in self.nodes:
            if self.nodes[node]["block"].type == "input":
                return node
        return -1

    @property
    def output_idx(self) -> int:
        for node in self.nodes:
            if self.nodes[node]["block"].type == "output":
                return node
        return -1

    @property
    def valid(self) -> bool:
        if self.input_idx == -1 or self.output_idx == -1:
            return False
        return nx.is_directed_acyclic_graph(self)
