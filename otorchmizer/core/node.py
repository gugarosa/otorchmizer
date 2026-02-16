"""Node class for tree-based optimization (Genetic Programming).

Tree structure remains in Python; terminal values are torch.Tensor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

import otorchmizer.utils.constant as c
import otorchmizer.utils.exception as e


class Node:
    """A binary-tree node for composing GP expression trees.

    Nodes are either TERMINAL (holding a tensor value) or FUNCTION
    (holding an operation name like SUM, MUL, EXP, etc.).
    """

    def __init__(
        self,
        name: Union[str, int],
        category: str,
        value: Optional[torch.Tensor] = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        parent: Optional[Node] = None,
    ) -> None:
        """Initialization method.

        Args:
            name: Node identifier (terminal index or function name).
            category: "TERMINAL" or "FUNCTION".
            value: Tensor value (terminals only).
            left: Left child node.
            right: Right child node.
            parent: Parent node.
        """

        self.name = name
        self.category = category
        self.value = value

        self.left = left
        self.right = right
        self.parent = parent

        self.flag = True

    def __repr__(self) -> str:
        return f"{self.category}:{self.name}:{self.flag}"

    def __str__(self) -> str:
        """Formatted tree display."""
        lines = _build_string(self)[0]
        return "\n" + "\n".join(lines)

    @property
    def name(self) -> Union[str, int]:
        return self._name

    @name.setter
    def name(self, name: Union[str, int]) -> None:
        if not isinstance(name, (str, int)):
            raise e.TypeError("`name` should be a string or integer")
        self._name = name

    @property
    def category(self) -> str:
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        if category not in ("TERMINAL", "FUNCTION"):
            raise e.ValueError("`category` should be 'TERMINAL' or 'FUNCTION'")
        self._category = category

    @property
    def value(self) -> Optional[torch.Tensor]:
        return self._value

    @value.setter
    def value(self, value: Optional[torch.Tensor]) -> None:
        if self.category != "TERMINAL":
            self._value = None
        else:
            if not isinstance(value, torch.Tensor):
                raise e.TypeError("`value` should be a torch.Tensor")
            self._value = value

    @property
    def left(self) -> Optional[Node]:
        return self._left

    @left.setter
    def left(self, left: Optional[Node]) -> None:
        if left is not None and not isinstance(left, Node):
            raise e.TypeError("`left` should be a Node")
        self._left = left

    @property
    def right(self) -> Optional[Node]:
        return self._right

    @right.setter
    def right(self, right: Optional[Node]) -> None:
        if right is not None and not isinstance(right, Node):
            raise e.TypeError("`right` should be a Node")
        self._right = right

    @property
    def parent(self) -> Optional[Node]:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional[Node]) -> None:
        if parent is not None and not isinstance(parent, Node):
            raise e.TypeError("`parent` should be a Node")
        self._parent = parent

    @property
    def flag(self) -> bool:
        return self._flag

    @flag.setter
    def flag(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise e.TypeError("`flag` should be a boolean")
        self._flag = flag

    @property
    def min_depth(self) -> int:
        return _properties(self)["min_depth"]

    @property
    def max_depth(self) -> int:
        return _properties(self)["max_depth"]

    @property
    def n_leaves(self) -> int:
        return _properties(self)["n_leaves"]

    @property
    def n_nodes(self) -> int:
        return _properties(self)["n_nodes"]

    @property
    def position(self) -> torch.Tensor:
        """Evaluates the expression tree and returns the result tensor."""
        return _evaluate(self)

    @property
    def post_order(self) -> List[Node]:
        """Post-order traversal of the tree."""

        result, stack = [], []
        node = self

        while True:
            while node is not None:
                if node.right is not None:
                    stack.append(node.right)
                stack.append(node)
                node = node.left

            node = stack.pop()

            if (node.right is not None and len(stack) > 0
                    and stack[-1] is node.right):
                stack.pop()
                stack.append(node)
                node = node.right
            else:
                result.append(node)
                node = None

            if len(stack) == 0:
                break

        return result

    @property
    def pre_order(self) -> List[Node]:
        """Pre-order traversal of the tree."""

        result, stack = [], [self]

        while stack:
            node = stack.pop()
            result.append(node)

            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

        return result

    def find_node(self, position: int) -> tuple[Optional[Node], bool]:
        """Finds a node at a given pre-order position.

        Args:
            position: Pre-order index.

        Returns:
            Tuple of (parent_node, is_left_child_flag).
        """

        pre_order = self.pre_order
        if len(pre_order) > position:
            node = pre_order[position]

            if node.category == "TERMINAL":
                return node.parent, node.flag

            if node.category == "FUNCTION":
                if node.parent and node.parent.parent:
                    return node.parent.parent, node.parent.flag
                return None, False

        return None, False


def _evaluate(node: Optional[Node]) -> Optional[torch.Tensor]:
    """Recursively evaluates a node tree using PyTorch operations."""

    if node is None:
        return None

    x = _evaluate(node.left)
    y = _evaluate(node.right)

    if node.category == "TERMINAL":
        return node.value

    ops = {
        "SUM": lambda a, b: a + b,
        "SUB": lambda a, b: a - b,
        "MUL": lambda a, b: a * b,
        "DIV": lambda a, b: a / (b + c.EPSILON),
        "EXP": lambda a, _: torch.exp(a),
        "SQRT": lambda a, _: torch.sqrt(torch.abs(a)),
        "LOG": lambda a, _: torch.log(torch.abs(a) + c.EPSILON),
        "ABS": lambda a, _: torch.abs(a),
        "SIN": lambda a, _: torch.sin(a),
        "COS": lambda a, _: torch.cos(a),
    }

    if node.name in ops:
        return ops[node.name](x, y)

    return None


def _properties(node: Node) -> Dict[str, Any]:
    """Computes tree properties via BFS."""

    min_depth, max_depth = 0, -1
    n_leaves = n_nodes = 0

    nodes = [node]
    while nodes:
        max_depth += 1
        next_nodes = []

        for n in nodes:
            n_nodes += 1
            if n.left is None and n.right is None:
                if min_depth == 0:
                    min_depth = max_depth
                n_leaves += 1
            if n.left is not None:
                next_nodes.append(n.left)
            if n.right is not None:
                next_nodes.append(n.right)

        nodes = next_nodes

    return {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
    }


def _build_string(node: Optional[Node]) -> tuple:
    """Builds a formatted string for displaying the nodes.

    References:
        https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py#L153

    Args:
        node: An instance of the Node class (can be a tree of Nodes).

    Returns:
        Tuple of (lines, width, start, end) for formatted display.
    """

    if node is None:
        return [], 0, 0, 0

    first_line, second_line = [], []

    name = str(node.name)
    gap = width = len(name)

    left_branch, left_width, left_start, left_end = _build_string(node.left)
    right_branch, right_width, right_start, right_end = _build_string(node.right)

    if left_width > 0:
        left = (left_start + left_end) // 2 + 1

        first_line.append(" " * (left + 1))
        first_line.append("_" * (left_width - left))

        second_line.append(" " * left + "/")
        second_line.append(" " * (left_width - left))

        start = left_width + 1
        gap += 1
    else:
        start = 0

    first_line.append(name)
    second_line.append(" " * width)

    if right_width > 0:
        right = (right_start + right_end) // 2

        first_line.append("_" * right)
        first_line.append(" " * (right_width - right + 1))

        second_line.append(" " * right + "\\")
        second_line.append(" " * (right_width - right))

        gap += 1

    end = start + width - 1
    gap = " " * gap

    lines = ["".join(first_line), "".join(second_line)]

    for i in range(max(len(left_branch), len(right_branch))):
        if i < len(left_branch):
            left_line = left_branch[i]
        else:
            left_line = " " * left_width

        if i < len(right_branch):
            right_line = right_branch[i]
        else:
            right_line = " " * right_width

        lines.append(left_line + gap + right_line)

    return lines, len(lines[0]), start, end
