"""Tests for new core features: multi-GPU, mixed-precision, CUDA Graphs, torch.compile."""

import pytest
import torch

from otorchmizer.core.device import CUDAGraphRunner, DeviceManager
from otorchmizer.core.population import Population
from otorchmizer.core.optimizer import Optimizer, UpdateContext
from otorchmizer.core.function import Function
from otorchmizer.core.space import Space


# ─── DeviceManager new features ───────────────────────────────────

class TestDeviceManagerDtype:
    def test_default_dtype(self):
        dm = DeviceManager("cpu")
        assert dm.dtype == torch.float32

    def test_custom_dtype(self):
        dm = DeviceManager("cpu", dtype=torch.float64)
        assert dm.dtype == torch.float64

    def test_zeros_inherits_dtype(self):
        dm = DeviceManager("cpu", dtype=torch.float64)
        t = dm.zeros(3)
        assert t.dtype == torch.float64

    def test_full_inherits_dtype(self):
        dm = DeviceManager("cpu", dtype=torch.float64)
        t = dm.full((3,), 1.0)
        assert t.dtype == torch.float64

    def test_tensor_inherits_dtype(self):
        dm = DeviceManager("cpu", dtype=torch.float64)
        t = dm.tensor([1.0, 2.0])
        assert t.dtype == torch.float64

    def test_dtype_override(self):
        dm = DeviceManager("cpu", dtype=torch.float64)
        t = dm.zeros(3, dtype=torch.float32)
        assert t.dtype == torch.float32


class TestDeviceManagerAvailableGPUs:
    def test_returns_list(self):
        gpus = DeviceManager.available_gpus()
        assert isinstance(gpus, list)
        # On CPU-only, this returns an empty list

    def test_each_entry_is_device(self):
        for dev in DeviceManager.available_gpus():
            assert isinstance(dev, torch.device)


class TestDeviceManagerScatterGather:
    def test_scatter_splits_evenly(self):
        t = torch.arange(12).reshape(4, 3).float()
        devices = [torch.device("cpu"), torch.device("cpu")]
        chunks = DeviceManager.scatter(t, devices)
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 3)
        assert chunks[1].shape == (2, 3)

    def test_gather_reconstructs(self):
        t = torch.arange(12).reshape(4, 3).float()
        devices = [torch.device("cpu"), torch.device("cpu")]
        chunks = DeviceManager.scatter(t, devices)
        merged = DeviceManager.gather(chunks, torch.device("cpu"))
        assert torch.equal(merged, t)

    def test_scatter_uneven(self):
        t = torch.arange(15).reshape(5, 3).float()
        devices = [torch.device("cpu"), torch.device("cpu")]
        chunks = DeviceManager.scatter(t, devices)
        assert chunks[0].shape[0] + chunks[1].shape[0] == 5


class TestDeviceManagerAutocast:
    def test_autocast_cpu_noop(self):
        dm = DeviceManager("cpu")
        with dm.autocast():
            t = torch.randn(3, 3) @ torch.randn(3, 3)
        # No error — CPU autocast is a no-op in our implementation
        assert t.shape == (3, 3)

    def test_autocast_disabled(self):
        dm = DeviceManager("cpu")
        with dm.autocast(enabled=False):
            t = torch.randn(3) + 1
        assert t.shape == (3,)


class TestCUDAGraphsAvailability:
    def test_supports_cuda_graphs(self):
        result = DeviceManager.supports_cuda_graphs()
        assert isinstance(result, bool)

    def test_capture_requires_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available — skip negative test")
        with pytest.raises(RuntimeError, match="CUDA"):
            DeviceManager.capture_graph(lambda x: x.add_(1), torch.randn(3))


# ─── Population new features ──────────────────────────────────────

class TestPopulationDtype:
    def test_default_dtype(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(5, 1, 1, lb, ub)
        assert pop.dtype == torch.float32
        assert pop.positions.dtype == torch.float32

    def test_float64(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(5, 1, 1, lb, ub, dtype=torch.float64)
        assert pop.dtype == torch.float64
        assert pop.positions.dtype == torch.float64
        assert pop.fitness.dtype == torch.float64


class TestPopulationTo:
    def test_to_same_device(self):
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pop = Population(10, 2, 1, lb, ub)
        pop.initialize_uniform()
        pop.to(torch.device("cpu"))
        assert pop.device == torch.device("cpu")

    def test_to_changes_dtype(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(5, 1, 1, lb, ub)
        pop.initialize_uniform()
        pop.to(torch.device("cpu"), dtype=torch.float64)
        assert pop.dtype == torch.float64
        assert pop.positions.dtype == torch.float64
        assert pop.fitness.dtype == torch.float64
        assert pop.lb.dtype == torch.float64

    def test_to_returns_self(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(5, 1, 1, lb, ub)
        result = pop.to(torch.device("cpu"))
        assert result is pop


class TestPopulationScatterGather:
    def test_scatter_into_two(self):
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([5.0, 5.0])
        pop = Population(10, 2, 1, lb, ub)
        pop.initialize_uniform()
        pop.fitness = torch.arange(10, dtype=torch.float32)

        devices = [torch.device("cpu"), torch.device("cpu")]
        subs = pop.scatter(devices)
        assert len(subs) == 2
        assert subs[0].n_agents == 5
        assert subs[1].n_agents == 5
        assert subs[0].n_variables == 2

    def test_gather_merges(self):
        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([5.0, 5.0])
        pop = Population(10, 2, 1, lb, ub)
        pop.initialize_uniform()
        pop.fitness = torch.arange(10, dtype=torch.float32)
        pop.update_best()

        devices = [torch.device("cpu"), torch.device("cpu")]
        subs = pop.scatter(devices)
        merged = Population.gather(subs, torch.device("cpu"))
        assert merged.n_agents == 10
        assert merged.positions.shape == (10, 2, 1)
        assert merged.fitness.shape == (10,)

    def test_gather_finds_global_best(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([10.0])
        pop = Population(6, 1, 1, lb, ub)
        pop.positions = torch.tensor([[[5.0]], [[3.0]], [[7.0]], [[1.0]], [[9.0]], [[2.0]]])
        pop.fitness = torch.tensor([5.0, 3.0, 7.0, 1.0, 9.0, 2.0])
        pop.update_best()

        devices = [torch.device("cpu"), torch.device("cpu")]
        subs = pop.scatter(devices)
        # Update each sub-population's best
        for s in subs:
            s.update_best()

        merged = Population.gather(subs, torch.device("cpu"))
        assert merged.best_fitness.item() == pytest.approx(1.0)

    def test_scatter_preserves_bounds(self):
        lb = torch.tensor([-5.0, -3.0])
        ub = torch.tensor([5.0, 3.0])
        pop = Population(8, 2, 1, lb, ub)
        pop.initialize_uniform()

        subs = pop.scatter([torch.device("cpu"), torch.device("cpu")])
        for s in subs:
            assert s.lb.squeeze().tolist() == pytest.approx([-5.0, -3.0])
            assert s.ub.squeeze().tolist() == pytest.approx([5.0, 3.0])


class TestPopulationRepr:
    def test_repr_includes_dtype(self):
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        pop = Population(3, 1, 1, lb, ub, dtype=torch.float64)
        r = repr(pop)
        assert "float64" in r


# ─── Optimizer new features ───────────────────────────────────────

class TestOptimizerCallDispatch:
    def test_call_dispatches_to_update(self):
        """__call__ should invoke update when no compiled update exists."""
        calls = []

        class DummyOpt(Optimizer):
            def update(self, ctx):
                calls.append("update")

        opt = DummyOpt()
        ctx = UpdateContext(None, None, 0, 10, torch.device("cpu"))
        opt(ctx)
        assert calls == ["update"]

    def test_compiled_update_attribute(self):
        opt = Optimizer()
        assert opt._compiled_update is None

    def test_repr_shows_compiled(self):
        opt = Optimizer()
        assert "compiled" not in repr(opt)
        opt._compiled_update = lambda ctx: None
        assert "compiled=True" in repr(opt)


class TestTorchCompile:
    def test_torch_compile_sets_attribute(self):
        class SimpleOpt(Optimizer):
            def update(self, ctx):
                pass

        opt = SimpleOpt()
        opt.torch_compile()
        assert opt._compiled_update is not None

    def test_compiled_call_invokes_compiled(self):
        calls = []

        class TrackOpt(Optimizer):
            def update(self, ctx):
                calls.append("original")

        opt = TrackOpt()
        # After torch_compile, __call__ should use compiled path
        opt.torch_compile()
        ctx = UpdateContext(None, None, 0, 10, torch.device("cpu"))
        opt(ctx)
        # The compiled version still calls the same update logic
        # but through torch.compile wrapper
        assert len(calls) >= 1


# ─── Integration: end-to-end with new features ───────────────────

class TestIntegrationNewFeatures:
    def test_population_dtype_in_optimizer(self):
        """Optimizer works with float64 population."""
        from otorchmizer.optimizers.swarm import PSO

        fn = Function(lambda x: (x ** 2).sum(dim=(-1, -2)))
        lb = torch.zeros(3)
        ub = torch.ones(3) * 5
        space = Space(n_agents=10, n_variables=3, lower_bound=lb, upper_bound=ub)
        space.population = Population(10, 3, 1, lb, ub, dtype=torch.float64)
        space.population.initialize_uniform()

        opt = PSO()
        opt.compile(space.population)
        opt.evaluate(space.population, fn)

        ctx = UpdateContext(space, fn, 0, 10, torch.device("cpu"))
        opt.update(ctx)
        space.population.clip()
        opt.evaluate(space.population, fn)

        assert space.population.best_fitness.item() < 100

    def test_scatter_gather_roundtrip(self):
        """Population survives scatter→update→gather cycle."""
        lb = torch.zeros(3)
        ub = torch.ones(3) * 5
        pop = Population(20, 3, 1, lb, ub)
        pop.initialize_uniform()
        pop.fitness = torch.rand(20)
        pop.update_best()

        original_best = pop.best_fitness.item()

        devices = [torch.device("cpu"), torch.device("cpu")]
        subs = pop.scatter(devices)
        for s in subs:
            s.update_best()

        merged = Population.gather(subs, torch.device("cpu"))
        assert merged.n_agents == 20
        assert merged.best_fitness.item() <= original_best + 1e-6
