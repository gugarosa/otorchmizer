"""Tests for core modules: Population, Space, Function, Optimizer."""

import pytest
import torch

from otorchmizer.core.agent_view import AgentView
from otorchmizer.core.device import DeviceManager
from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.core.space import Space


class TestDeviceManager:
    def test_auto_resolves_cpu(self):
        dm = DeviceManager("cpu")
        assert dm.device == torch.device("cpu")

    def test_zeros(self):
        dm = DeviceManager("cpu")
        t = dm.zeros(3, 4)
        assert t.shape == (3, 4)
        assert t.device == torch.device("cpu")

    def test_rand(self):
        dm = DeviceManager("cpu")
        t = dm.rand(5)
        assert t.shape == (5,)

    def test_tensor(self):
        dm = DeviceManager("cpu")
        t = dm.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)


class TestPopulation:
    def test_creation(self):
        lb = torch.tensor([-1.0, -2.0])
        ub = torch.tensor([1.0, 2.0])
        pop = Population(10, 2, 1, lb, ub)
        assert pop.positions.shape == (10, 2, 1)
        assert pop.fitness.shape == (10,)
        assert pop.n_agents == 10

    def test_initialize_uniform(self):
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pop = Population(100, 2, 1, lb, ub)
        pop.initialize_uniform()
        assert (pop.positions >= 0).all()
        assert (pop.positions <= 1).all()

    def test_initialize_binary(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(50, 1, 1, lb, ub)
        pop.initialize_binary()
        unique_vals = pop.positions.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_clip(self):
        lb = torch.tensor([-1.0])
        ub = torch.tensor([1.0])
        pop = Population(5, 1, 1, lb, ub)
        pop.positions = torch.tensor([[[5.0]], [[-3.0]], [[0.5]], [[1.0]], [[-1.0]]])
        pop.clip()
        assert pop.positions.max() <= 1.0
        assert pop.positions.min() >= -1.0

    def test_update_best(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(3, 1, 1, lb, ub)
        pop.fitness = torch.tensor([3.0, 1.0, 2.0])
        pop.positions = torch.tensor([[[0.3]], [[0.1]], [[0.2]]])
        pop.update_best()
        assert pop.best_fitness.item() == 1.0
        assert pop.best_position.item() == pytest.approx(0.1)

    def test_sort_by_fitness(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(3, 1, 1, lb, ub)
        pop.fitness = torch.tensor([3.0, 1.0, 2.0])
        pop.positions = torch.tensor([[[0.3]], [[0.1]], [[0.2]]])
        pop.sort_by_fitness()
        assert pop.fitness.tolist() == [1.0, 2.0, 3.0]

    def test_clone_positions(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(3, 1, 1, lb, ub)
        pop.initialize_uniform()
        cloned = pop.clone_positions()
        assert torch.equal(cloned, pop.positions)
        cloned[0] = 999.0
        assert not torch.equal(cloned, pop.positions)


class TestAgentView:
    def test_view_access(self):
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pop = Population(5, 2, 1, lb, ub)
        pop.initialize_uniform()
        pop.fitness = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        view = AgentView(pop, 1)
        assert view.fit == 2.0
        assert view.position.shape == (2, 1)
        assert view.n_variables == 2


class TestFunction:
    def test_single_agent_function(self):
        fn = Function(lambda x: (x ** 2).sum())
        positions = torch.rand(10, 3, 1)
        result = fn(positions)
        assert result.shape == (10,)

    def test_batch_function(self):
        def batch_fn(positions):
            return (positions ** 2).sum(dim=(1, 2))
        fn = Function(batch_fn, batch=True)
        positions = torch.rand(10, 3, 1)
        result = fn(positions)
        assert result.shape == (10,)

    def test_name_detection(self):
        def sphere(x):
            return (x ** 2).sum()
        fn = Function(sphere)
        assert fn.name == "sphere"


class TestSpace:
    def test_build(self):
        space = Space(n_agents=10, n_variables=3, lower_bound=[-1, -1, -1],
                      upper_bound=[1, 1, 1], device="cpu")
        space.build()
        assert space.built
        assert space.population.positions.shape == (10, 3, 1)

    def test_clip(self):
        space = Space(n_agents=5, n_variables=2, lower_bound=[0, 0],
                      upper_bound=[1, 1], device="cpu")
        space.build()
        space.population.positions.fill_(5.0)
        space.clip()
        assert space.population.positions.max() <= 1.0


class TestOptimizer:
    def test_base_optimizer_build(self):
        opt = Optimizer(params={"test_param": 42})
        assert opt.built
        assert opt.test_param == 42
        assert opt.algorithm == "Optimizer"

    def test_base_evaluate(self):
        fn = Function(lambda x: (x ** 2).sum())
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        pop = Population(10, 2, 1, lb, ub)
        pop.initialize_uniform()

        opt = Optimizer()
        opt.evaluate(pop, fn)
        assert pop.fitness.shape == (10,)
        assert pop.best_fitness < float("inf")

    def test_update_not_implemented(self):
        opt = Optimizer()
        with pytest.raises(NotImplementedError):
            opt.update(UpdateContext(None, None, 0, 100, torch.device("cpu")))
