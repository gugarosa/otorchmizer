"""Tests for core.node — binary tree nodes for GP expression trees."""

import pytest
import torch

from otorchmizer.core.node import Node
import otorchmizer.utils.exception as e


class TestNodeCreation:
    def test_terminal_node(self):
        val = torch.tensor([1.0, 2.0])
        node = Node(0, "TERMINAL", value=val)
        assert node.name == 0
        assert node.category == "TERMINAL"
        assert torch.equal(node.value, val)
        assert node.left is None
        assert node.right is None
        assert node.parent is None
        assert node.flag is True

    def test_function_node(self):
        node = Node("SUM", "FUNCTION")
        assert node.name == "SUM"
        assert node.category == "FUNCTION"
        assert node.value is None

    def test_function_node_ignores_value(self):
        node = Node("MUL", "FUNCTION", value=torch.tensor([1.0]))
        assert node.value is None

    def test_repr(self):
        node = Node("SUM", "FUNCTION")
        assert repr(node) == "FUNCTION:SUM:True"

    def test_str_single_node(self):
        val = torch.tensor([1.0])
        node = Node(0, "TERMINAL", value=val)
        s = str(node)
        assert "0" in s


class TestNodeValidation:
    def test_invalid_name_type(self):
        with pytest.raises(e.TypeError):
            Node(3.14, "TERMINAL", value=torch.tensor([1.0]))

    def test_invalid_category(self):
        with pytest.raises(e.ValueError):
            Node("x", "INVALID", value=torch.tensor([1.0]))

    def test_terminal_requires_tensor_value(self):
        with pytest.raises(e.TypeError):
            Node(0, "TERMINAL", value=42)

    def test_invalid_left_child(self):
        node = Node("SUM", "FUNCTION")
        with pytest.raises(e.TypeError):
            node.left = "not_a_node"

    def test_invalid_right_child(self):
        node = Node("SUM", "FUNCTION")
        with pytest.raises(e.TypeError):
            node.right = "not_a_node"

    def test_invalid_parent(self):
        node = Node(0, "TERMINAL", value=torch.tensor([1.0]))
        with pytest.raises(e.TypeError):
            node.parent = "not_a_node"

    def test_invalid_flag(self):
        node = Node("SUM", "FUNCTION")
        with pytest.raises(e.TypeError):
            node.flag = "true"


class TestTreeStructure:
    @pytest.fixture
    def simple_tree(self):
        """Creates: SUM(terminal_0, terminal_1)"""
        root = Node("SUM", "FUNCTION")
        left = Node(0, "TERMINAL", value=torch.tensor([3.0]))
        right = Node(1, "TERMINAL", value=torch.tensor([4.0]))
        root.left = left
        root.right = right
        left.parent = root
        right.parent = root
        right.flag = False
        return root

    def test_tree_properties(self, simple_tree):
        assert simple_tree.n_nodes == 3
        assert simple_tree.n_leaves == 2
        assert simple_tree.min_depth == 1
        assert simple_tree.max_depth == 1

    def test_pre_order(self, simple_tree):
        pre = simple_tree.pre_order
        assert len(pre) == 3
        assert pre[0].name == "SUM"
        assert pre[1].category == "TERMINAL"
        assert pre[2].category == "TERMINAL"

    def test_post_order(self, simple_tree):
        post = simple_tree.post_order
        assert len(post) == 3
        assert post[-1].name == "SUM"

    def test_position_evaluates_sum(self, simple_tree):
        result = simple_tree.position
        assert torch.allclose(result, torch.tensor([7.0]))

    def test_find_node_terminal(self, simple_tree):
        parent, flag = simple_tree.find_node(1)
        assert parent is simple_tree

    def test_find_node_out_of_range(self, simple_tree):
        parent, flag = simple_tree.find_node(100)
        assert parent is None
        assert flag is False


class TestTreeEvaluation:
    def _make_unary(self, op: str, val: torch.Tensor):
        root = Node(op, "FUNCTION")
        child = Node(0, "TERMINAL", value=val)
        root.left = child
        child.parent = root
        return root

    def _make_binary(self, op: str, left_val: torch.Tensor, right_val: torch.Tensor):
        root = Node(op, "FUNCTION")
        left = Node(0, "TERMINAL", value=left_val)
        right = Node(1, "TERMINAL", value=right_val)
        root.left = left
        root.right = right
        left.parent = root
        right.parent = root
        return root

    def test_sub(self):
        tree = self._make_binary("SUB", torch.tensor([10.0]), torch.tensor([3.0]))
        assert torch.allclose(tree.position, torch.tensor([7.0]))

    def test_mul(self):
        tree = self._make_binary("MUL", torch.tensor([3.0]), torch.tensor([4.0]))
        assert torch.allclose(tree.position, torch.tensor([12.0]))

    def test_div(self):
        tree = self._make_binary("DIV", torch.tensor([6.0]), torch.tensor([2.0]))
        assert torch.allclose(tree.position, torch.tensor([3.0]), atol=1e-6)

    def test_div_by_zero_safe(self):
        tree = self._make_binary("DIV", torch.tensor([6.0]), torch.tensor([0.0]))
        result = tree.position
        assert torch.isfinite(result).all()

    def test_exp(self):
        tree = self._make_unary("EXP", torch.tensor([0.0]))
        assert torch.allclose(tree.position, torch.tensor([1.0]))

    def test_sqrt(self):
        tree = self._make_unary("SQRT", torch.tensor([4.0]))
        assert torch.allclose(tree.position, torch.tensor([2.0]))

    def test_abs(self):
        tree = self._make_unary("ABS", torch.tensor([-5.0]))
        assert torch.allclose(tree.position, torch.tensor([5.0]))

    def test_sin(self):
        tree = self._make_unary("SIN", torch.tensor([0.0]))
        assert torch.allclose(tree.position, torch.tensor([0.0]), atol=1e-6)

    def test_cos(self):
        tree = self._make_unary("COS", torch.tensor([0.0]))
        assert torch.allclose(tree.position, torch.tensor([1.0]), atol=1e-6)

    def test_log(self):
        tree = self._make_unary("LOG", torch.tensor([1.0]))
        assert torch.allclose(tree.position, torch.tensor([0.0]), atol=1e-6)


class TestDeepTree:
    def test_depth_3_tree(self):
        """Build SUM(MUL(2, 3), 4) = 10"""
        mul_node = Node("MUL", "FUNCTION")
        t2 = Node(0, "TERMINAL", value=torch.tensor([2.0]))
        t3 = Node(1, "TERMINAL", value=torch.tensor([3.0]))
        mul_node.left = t2
        mul_node.right = t3
        t2.parent = mul_node
        t3.parent = mul_node

        root = Node("SUM", "FUNCTION")
        t4 = Node(2, "TERMINAL", value=torch.tensor([4.0]))
        root.left = mul_node
        root.right = t4
        mul_node.parent = root
        t4.parent = root

        assert root.n_nodes == 5
        assert root.n_leaves == 3
        assert root.max_depth == 2
        assert torch.allclose(root.position, torch.tensor([10.0]))
