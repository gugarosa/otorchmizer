"""Tests for specialized search spaces."""

import pytest
import torch

from otorchmizer.spaces.boolean import BooleanSpace
from otorchmizer.spaces.graph import GraphSpace
from otorchmizer.spaces.grid import GridSpace
from otorchmizer.spaces.hyper_complex import HyperComplexSpace
from otorchmizer.spaces.pareto import ParetoSpace
from otorchmizer.spaces.tree import TreeSpace
import otorchmizer.utils.exception as e


class TestBooleanSpace:
    def test_creation(self):
        space = BooleanSpace(n_agents=10, n_variables=5)
        assert space.built
        assert space.population.n_agents == 10
        assert space.population.n_variables == 5

    def test_binary_positions(self):
        space = BooleanSpace(n_agents=20, n_variables=3)
        vals = space.population.positions.unique()
        assert all(v in [0.0, 1.0] for v in vals.tolist())

    def test_bounds_fixed(self):
        space = BooleanSpace(n_agents=5, n_variables=2)
        assert space.population.lb.max().item() == 0.0
        assert space.population.ub.min().item() == 1.0

    def test_mapping(self):
        space = BooleanSpace(n_agents=5, n_variables=2, mapping=["x", "y"])
        assert space.population.mapping == ["x", "y"]


class TestGraphSpace:
    def test_creation(self):
        space = GraphSpace(n_blocks=5)
        assert space.n_blocks == 5
        assert space.built


class TestGridSpace:
    def test_creation(self):
        space = GridSpace(
            n_variables=2,
            step=[1.0, 1.0],
            lower_bound=[0.0, 0.0],
            upper_bound=[2.0, 2.0],
        )
        assert space.built
        # 3 values per dim (0,1,2) → 9 grid points
        assert space.population.n_agents == 9

    def test_grid_positions(self):
        space = GridSpace(
            n_variables=1,
            step=0.5,
            lower_bound=[0.0],
            upper_bound=[1.0],
        )
        # Values should be 0.0, 0.5, 1.0 → 3 agents
        assert space.population.n_agents == 3
        positions = space.population.positions.squeeze().cpu()
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(positions, expected)

    def test_single_step(self):
        """Scalar step broadcast to all variables."""
        space = GridSpace(
            n_variables=2,
            step=1.0,
            lower_bound=[0.0, 0.0],
            upper_bound=[1.0, 1.0],
        )
        # 2 values per dim → 4 grid points
        assert space.population.n_agents == 4


class TestHyperComplexSpace:
    def test_creation(self):
        space = HyperComplexSpace(n_agents=10, n_variables=3, n_dimensions=4)
        assert space.built
        assert space.population.n_agents == 10
        assert space.population.n_variables == 3
        assert space.population.n_dimensions == 4

    def test_bounds_unit(self):
        space = HyperComplexSpace(n_agents=5, n_variables=2, n_dimensions=2)
        assert space.population.lb.max().item() == 0.0
        assert space.population.ub.min().item() == 1.0

    def test_positions_in_range(self):
        space = HyperComplexSpace(n_agents=20, n_variables=2, n_dimensions=4)
        assert (space.population.positions >= 0).all()
        assert (space.population.positions <= 1).all()


class TestParetoSpace:
    def test_creation(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        space = ParetoSpace(data_points=data)
        assert space.built
        assert space.population.n_agents == 3
        assert space.population.n_variables == 2

    def test_positions_match_data(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        space = ParetoSpace(data_points=data)
        pos = space.population.positions.squeeze(-1).cpu()
        assert torch.allclose(pos, data)

    def test_clip_is_noop(self):
        data = torch.tensor([[100.0, 200.0]])
        space = ParetoSpace(data_points=data)
        space.clip()
        pos = space.population.positions.squeeze(-1).cpu()
        assert torch.allclose(pos, data)


class TestTreeSpace:
    def test_creation(self):
        space = TreeSpace(
            n_agents=5,
            n_variables=2,
            lower_bound=[-1.0, -1.0],
            upper_bound=[1.0, 1.0],
            n_terminals=2,
            min_depth=1,
            max_depth=2,
            functions=["SUM", "MUL"],
        )
        assert space.built
        assert len(space.trees) == 5
        assert space.best_tree is not None

    def test_terminals_created(self):
        space = TreeSpace(
            n_agents=3,
            n_variables=2,
            lower_bound=[0.0, 0.0],
            upper_bound=[1.0, 1.0],
            n_terminals=3,
            min_depth=1,
            max_depth=2,
            functions=["SUM"],
        )
        assert len(space.terminals) == 3

    def test_grow_returns_node(self):
        space = TreeSpace(
            n_agents=2,
            n_variables=1,
            lower_bound=[0.0],
            upper_bound=[1.0],
            n_terminals=1,
            min_depth=1,
            max_depth=3,
            functions=["SUM", "SUB"],
        )
        from otorchmizer.core.node import Node
        tree = space.grow(1, 3)
        assert isinstance(tree, Node)

    def test_invalid_n_terminals(self):
        with pytest.raises(e.ValueError):
            TreeSpace(
                n_agents=2, n_variables=1,
                lower_bound=[0.0], upper_bound=[1.0],
                n_terminals=0, min_depth=1, max_depth=2,
                functions=["SUM"],
            )

    def test_invalid_min_depth(self):
        with pytest.raises(e.ValueError):
            TreeSpace(
                n_agents=2, n_variables=1,
                lower_bound=[0.0], upper_bound=[1.0],
                n_terminals=1, min_depth=0, max_depth=2,
                functions=["SUM"],
            )

    def test_invalid_max_depth_less_than_min(self):
        with pytest.raises(e.ValueError):
            TreeSpace(
                n_agents=2, n_variables=1,
                lower_bound=[0.0], upper_bound=[1.0],
                n_terminals=1, min_depth=3, max_depth=1,
                functions=["SUM"],
            )
