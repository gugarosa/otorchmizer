"""Integration test: end-to-end optimization with Otorchmizer."""

import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function
from otorchmizer.optimizers.swarm import PSO, WOA
from otorchmizer.optimizers.evolutionary import GA
from otorchmizer.spaces import SearchSpace


def _sphere(x):
    return (x ** 2).sum()


class TestOtorchmizerIntegration:
    def test_pso_full_run(self):
        """Full PSO optimization should complete without errors."""

        torch.manual_seed(42)

        space = SearchSpace(
            n_agents=20, n_variables=2,
            lower_bound=[-10, -10], upper_bound=[10, 10],
            device="cpu",
        )
        optimizer = PSO()
        function = Function(_sphere)

        opt = Otorchmizer(space, optimizer, function)
        opt.start(n_iterations=50)

        assert opt.space.best_fitness.item() < 100.0
        assert hasattr(opt.history, "best_agent")

    def test_woa_full_run(self):
        torch.manual_seed(42)

        space = SearchSpace(
            n_agents=20, n_variables=2,
            lower_bound=[-10, -10], upper_bound=[10, 10],
            device="cpu",
        )
        optimizer = WOA()
        function = Function(_sphere)

        opt = Otorchmizer(space, optimizer, function)
        opt.start(n_iterations=50)

        assert opt.space.best_fitness.item() < 100.0

    def test_ga_full_run(self):
        torch.manual_seed(42)

        space = SearchSpace(
            n_agents=30, n_variables=2,
            lower_bound=[-10, -10], upper_bound=[10, 10],
            device="cpu",
        )
        optimizer = GA()
        function = Function(_sphere)

        opt = Otorchmizer(space, optimizer, function)
        opt.start(n_iterations=50)

        assert opt.space.best_fitness.item() < 100.0

    def test_custom_params(self):
        """Optimizer params should be overridable."""

        pso = PSO(params={"w": 0.5, "c1": 2.0, "c2": 2.0})
        assert pso.w == 0.5
        assert pso.c1 == 2.0
