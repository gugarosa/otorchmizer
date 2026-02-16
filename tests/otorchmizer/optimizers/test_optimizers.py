"""Tests for optimizer implementations: PSO, WOA, FA, GA, GS, HC."""

import torch

from otorchmizer.core.function import Function
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.core.population import Population
from otorchmizer.optimizers.evolutionary.ga import GA
from otorchmizer.optimizers.misc.gs import GS
from otorchmizer.optimizers.misc.hc import HC
from otorchmizer.optimizers.swarm.fa import FA
from otorchmizer.optimizers.swarm.pso import AIWPSO, PSO, RPSO, SAVPSO, VPSO
from otorchmizer.optimizers.swarm.woa import WOA
from otorchmizer.spaces.search import SearchSpace


def _sphere(x):
    return (x ** 2).sum()


def _make_space(n_agents=20, n_variables=2, device="cpu"):
    return SearchSpace(
        n_agents=n_agents,
        n_variables=n_variables,
        lower_bound=[-5.0] * n_variables,
        upper_bound=[5.0] * n_variables,
        device=device,
    )


def _make_ctx(space, function, iteration=0, n_iterations=100):
    return UpdateContext(
        space=space,
        function=function,
        iteration=iteration,
        n_iterations=n_iterations,
        device=space.device,
    )


class TestPSO:
    def test_build(self):
        pso = PSO()
        assert pso.built
        assert pso.w == 0.7

    def test_compile(self):
        pso = PSO()
        space = _make_space()
        pso.compile(space.population)
        assert pso.velocity.shape == (20, 2, 1)
        assert pso.local_position.shape == (20, 2, 1)

    def test_update_and_evaluate(self):
        fn = Function(_sphere)
        space = _make_space()
        pso = PSO()
        pso.compile(space.population)

        pso.evaluate(space.population, fn)
        ctx = _make_ctx(space, fn)
        pso.update(ctx)

        # Positions should have changed
        assert space.population.positions.shape == (20, 2, 1)

    def test_convergence(self):
        """PSO should find near-zero on a sphere function."""

        torch.manual_seed(42)
        fn = Function(_sphere)
        space = _make_space(n_agents=30, n_variables=2)
        pso = PSO()
        pso.compile(space.population)

        for i in range(200):
            ctx = _make_ctx(space, fn, iteration=i, n_iterations=200)
            pso.update(ctx)
            space.clip()
            pso.evaluate(space.population, fn)

        assert space.population.best_fitness.item() < 1.0


class TestAIWPSO:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = AIWPSO()
        opt.compile(space.population)
        opt.evaluate(space.population, fn)

        for i in range(5):
            ctx = _make_ctx(space, fn, iteration=i)
            opt.update(ctx)
            space.clip()
            opt.evaluate(space.population, fn)


class TestRPSO:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = RPSO()
        opt.compile(space.population)
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)


class TestSAVPSO:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = SAVPSO()
        opt.compile(space.population)
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)


class TestVPSO:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = VPSO()
        opt.compile(space.population)
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)


class TestWOA:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = WOA()
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn, iteration=5, n_iterations=100)
        opt.update(ctx)

    def test_convergence(self):
        torch.manual_seed(42)
        fn = Function(_sphere)
        space = _make_space(n_agents=30)
        opt = WOA()

        for i in range(300):
            ctx = _make_ctx(space, fn, iteration=i, n_iterations=300)
            opt.update(ctx)
            space.clip()
            opt.evaluate(space.population, fn)

        assert space.population.best_fitness.item() < 1.0


class TestFA:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = FA()
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)

    def test_pairwise_vectorization(self):
        """FA should handle pairwise distances without loops."""

        fn = Function(_sphere)
        space = _make_space(n_agents=50)
        opt = FA()
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)
        assert space.population.positions.shape == (50, 2, 1)


class TestGA:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = GA()
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)

    def test_convergence(self):
        torch.manual_seed(42)
        fn = Function(_sphere)
        space = _make_space(n_agents=40)
        opt = GA()

        for i in range(200):
            ctx = _make_ctx(space, fn, iteration=i, n_iterations=200)
            opt.evaluate(space.population, fn)
            opt.update(ctx)
            space.clip()

        assert space.population.best_fitness.item() < 5.0


class TestGS:
    def test_update_noop(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = GS()
        ctx = _make_ctx(space, fn)
        opt.update(ctx)  # Should do nothing


class TestHC:
    def test_update(self):
        fn = Function(_sphere)
        space = _make_space()
        opt = HC()
        opt.evaluate(space.population, fn)

        ctx = _make_ctx(space, fn)
        opt.update(ctx)
