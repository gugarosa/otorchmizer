"""Tests for core.block — graph-based optimization building blocks."""

import pytest

from otorchmizer.core.block import Block, InputBlock, InnerBlock, OutputBlock, Cell
import otorchmizer.utils.exception as e


class TestBlock:
    def test_creation(self):
        b = Block("inner", lambda x: x * 2, n_input=1, n_output=1)
        assert b.type == "inner"
        assert b.n_input == 1
        assert b.n_output == 1

    def test_call(self):
        b = Block("inner", lambda x: x * 2, n_input=1, n_output=1)
        assert b(5) == 10

    def test_invalid_type(self):
        with pytest.raises(e.ValueError):
            Block("invalid", lambda x: x, n_input=1, n_output=1)

    def test_invalid_pointer(self):
        with pytest.raises(e.TypeError):
            Block("inner", "not_callable", n_input=1, n_output=1)

    def test_invalid_n_input(self):
        with pytest.raises(e.ValueError):
            Block("inner", lambda x: x, n_input=0, n_output=1)

    def test_invalid_n_input_type(self):
        with pytest.raises(e.ValueError):
            Block("inner", lambda x: x, n_input=1.5, n_output=1)

    def test_invalid_n_output(self):
        with pytest.raises(e.ValueError):
            Block("inner", lambda x: x, n_input=1, n_output=-1)


class TestSpecializedBlocks:
    def test_input_block(self):
        b = InputBlock(n_input=2, n_output=2)
        assert b.type == "input"
        result = b(1, 2)
        assert result == (1, 2)

    def test_inner_block(self):
        b = InnerBlock(pointer=lambda x: x + 10, n_input=1, n_output=1)
        assert b.type == "inner"
        assert b(5) == 15

    def test_output_block(self):
        b = OutputBlock(n_input=1, n_output=1)
        assert b.type == "output"
        result = b(42)
        assert result == (42,)


class TestCell:
    def test_simple_dag(self):
        inp = InputBlock(n_input=1, n_output=1)
        inner = InnerBlock(lambda x: x * 2, n_input=1, n_output=1)
        out = OutputBlock(n_input=1, n_output=1)

        cell = Cell([inp, inner, out], [(0, 1), (1, 2)])
        assert cell.input_idx == 0
        assert cell.output_idx == 2
        assert cell.valid

    def test_valid_cell_call(self):
        inp = InputBlock(n_input=1, n_output=1)
        inner = InnerBlock(lambda x: x * 3, n_input=1, n_output=1)
        out = OutputBlock(n_input=1, n_output=1)

        cell = Cell([inp, inner, out], [(0, 1), (1, 2)])
        result = cell(5)
        assert len(result) > 0

    def test_no_input_block_invalid(self):
        inner = InnerBlock(lambda x: x, n_input=1, n_output=1)
        out = OutputBlock(n_input=1, n_output=1)
        cell = Cell([inner, out], [(0, 1)])
        assert not cell.valid

    def test_no_output_block_invalid(self):
        inp = InputBlock(n_input=1, n_output=1)
        inner = InnerBlock(lambda x: x, n_input=1, n_output=1)
        cell = Cell([inp, inner], [(0, 1)])
        assert not cell.valid

    def test_invalid_cell_returns_empty(self):
        inner = InnerBlock(lambda x: x, n_input=1, n_output=1)
        cell = Cell([inner], [])
        result = cell(5)
        assert result == []

    def test_edge_arity_mismatch_ignored(self):
        inp = InputBlock(n_input=1, n_output=1)
        inner = InnerBlock(lambda x: x, n_input=2, n_output=1)
        cell = Cell([inp, inner], [(0, 1)])
        # edge should not be added since n_output(0)=1 != n_input(1)=2
        assert len(cell.edges) == 0
