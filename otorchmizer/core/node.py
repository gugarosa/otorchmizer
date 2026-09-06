# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Node class for tree-based optimization (Genetic Programming).

Tree structure remains in Python; terminal values are torch.Tensor.

"""

from __future__ import annotations

import torch


class Node:
    """A binary-tree node for composing GP expression trees.

    Notes:
        Nodes are either TERMINAL (holding a tensor value) or FUNCTION
        (holding an operation name such as SUM, MUL, or EXP).
        Tree display follows https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py#L153.

    """

    def __init__(
        self,
        name: str | int,
        category: str,
        value: torch.Tensor | None = None,
        left: Node | None = None,
        right: Node | None = None,
        parent: Node | None = None,
    ) -> None:
        """Initialize an expression node and its explicit tree links.

        Args:
            name: Node identifier (terminal index or function name).
            category: "TERMINAL" or "FUNCTION".
            value: Tensor value (terminals only).
            left: Left child node.
            right: Right child node.
            parent: Parent node.

        Raises:
            TypeError: A name, terminal value, or tree link has an unsupported type.
            ValueError: The category is neither "TERMINAL" nor "FUNCTION".

        Notes:
            Function nodes discard the supplied value. Links are assigned without updating reciprocal links.
            The child-side flag starts as True and can be changed separately.

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
    def name(self) -> str | int:
        """Terminal identifier or function operation name."""

        return self._name

    @name.setter
    def name(self, name: str | int) -> None:
        if not isinstance(name, (str, int)):
            raise TypeError("`name` should be a string or integer.")
        self._name = name

    @property
    def category(self) -> str:
        """Node role, either "TERMINAL" or "FUNCTION"."""

        return self._category

    @category.setter
    def category(self, category: str) -> None:
        if category not in ("TERMINAL", "FUNCTION"):
            raise ValueError("`category` should be 'TERMINAL' or 'FUNCTION'.")
        self._category = category

    @property
    def value(self) -> torch.Tensor | None:
        """Stored terminal tensor, or None for a function node."""

        return self._value

    @value.setter
    def value(self, value: torch.Tensor | None) -> None:
        if self.category != "TERMINAL":
            self._value = None
        else:
            if not isinstance(value, torch.Tensor):
                raise TypeError("`value` should be a torch.Tensor.")
            self._value = value

    @property
    def left(self) -> Node | None:
        """Left child, or None when no left child is linked."""

        return self._left

    @left.setter
    def left(self, left: Node | None) -> None:
        if left is not None and not isinstance(left, Node):
            raise TypeError("`left` should be a Node.")
        self._left = left

    @property
    def right(self) -> Node | None:
        """Right child, or None when no right child is linked."""

        return self._right

    @right.setter
    def right(self, right: Node | None) -> None:
        if right is not None and not isinstance(right, Node):
            raise TypeError("`right` should be a Node.")
        self._right = right

    @property
    def parent(self) -> Node | None:
        """Parent node, or None when no parent is linked."""

        return self._parent

    @parent.setter
    def parent(self, parent: Node | None) -> None:
        if parent is not None and not isinstance(parent, Node):
            raise TypeError("`parent` should be a Node.")
        self._parent = parent

    @property
    def flag(self) -> bool:
        """Stored child-side flag returned during crossover lookup."""

        return self._flag

    @flag.setter
    def flag(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise TypeError("`flag` should be a boolean.")
        self._flag = flag

    @property
    def min_depth(self) -> int:
        """Shortest distance in edges from this node to a leaf."""

        return _properties(self)["min_depth"]

    @property
    def max_depth(self) -> int:
        """Longest distance in edges from this node to a leaf."""

        return _properties(self)["max_depth"]

    @property
    def n_leaves(self) -> int:
        """Number of reachable nodes with no children."""

        return _properties(self)["n_leaves"]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in this subtree, including this node."""

        return _properties(self)["n_nodes"]

    @property
    def position(self) -> torch.Tensor | None:
        """Evaluated expression tensor, or None for an unknown operation."""

        return _evaluate(self)

    @property
    def post_order(self) -> list[Node]:
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

            if node.right is not None and len(stack) > 0 and stack[-1] is node.right:
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
    def pre_order(self) -> list[Node]:
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

    def clone(self) -> Node:
        """Clone this subtree with independent tensors and rebuilt parent links.

        Returns:
            Independently owned subtree root.

        """

        value = self.value.clone() if self.value is not None else None
        root = Node(self.name, self.category, value=value)

        if self.left is not None:
            root.left = self.left.clone()
            root.left.parent = root
            root.left.flag = True

        if self.right is not None:
            root.right = self.right.clone()
            root.right.parent = root
            root.right.flag = False

        return root

    def find_node(self, position: int) -> tuple[Node | None, bool]:
        """Find a crossover parent and child-side flag by pre-order index.

        Args:
            position: Pre-order index.

        Returns:
            Parent and child-side flag for a terminal, grandparent and parent flag for a function, or (None, False).

        Notes:
            Function nodes require both a parent and a grandparent to produce a usable result.

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


def _evaluate(node: Node | None) -> torch.Tensor | None:
    value, valid = _evaluate_node(node)
    if value is None:
        return None
    if not value.is_floating_point():
        return value

    return torch.where(valid, value, torch.full_like(value, torch.nan))


def _promoted_float_dtype(*values: torch.Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


def _evaluate_node(node: Node | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if node is None:
        return None, None

    x, x_valid = _evaluate_node(node.left)
    y, y_valid = _evaluate_node(node.right)

    if node.category == "TERMINAL":
        return node.value, torch.isfinite(node.value).all()

    if x is None:
        raise ValueError("`node.left` should be defined for a function node.")
    if node.name in ("SUM", "SUB", "MUL", "DIV") and y is None:
        raise ValueError("`node.right` should be defined for a binary function node.")

    if node.name == "SUM":
        result = x + y
    elif node.name == "SUB":
        result = x - y
    elif node.name == "MUL":
        result = x * y
    elif node.name == "DIV":
        dtype = _promoted_float_dtype(x, y)
        numerator = x.to(dtype=dtype)
        denominator = y.to(dtype=dtype)
        denominator = torch.where(
            torch.abs(denominator) > torch.finfo(dtype).tiny,
            denominator,
            torch.ones_like(denominator),
        )
        result = numerator / denominator
    elif node.name == "EXP":
        result = torch.exp(x)
    elif node.name == "SQRT":
        result = torch.sqrt(torch.abs(x))
    elif node.name == "LOG":
        dtype = _promoted_float_dtype(x)
        value = x.to(dtype=dtype)
        result = torch.log(torch.clamp(torch.abs(value), min=torch.finfo(dtype).tiny))
    elif node.name == "ABS":
        result = torch.abs(x)
    elif node.name == "SIN":
        result = torch.sin(x)
    elif node.name == "COS":
        result = torch.cos(x)
    else:
        raise ValueError(f"`node.name={node.name}` should identify a supported function.")

    valid = x_valid & torch.isfinite(result).all()
    if y_valid is not None:
        valid = valid & y_valid
    return result, valid


def _properties(node: Node) -> dict[str, int]:
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


def _build_string(node: Node | None) -> tuple:
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
