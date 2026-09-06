# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import logging
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import dill
import numpy as np
import pytest
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Optimizer, Population
from otorchmizer.functions import ConstrainedFunction
from otorchmizer.functions.multi_objective import MultiObjectiveWeightedFunction
from otorchmizer.spaces import BooleanSpace, HyperComplexSpace, ParetoSpace, SearchSpace, TreeSpace
from otorchmizer.utils.callback import Callback


class HalvingOptimizer(Optimizer):
    def compile(self, population):
        self.compilations = getattr(self, "compilations", 0) + 1
        self.buffer = population.positions.new_tensor([3.0])

    def update(self, ctx):
        ctx.space.population.positions.mul_(0.5)
        self.seen = ctx.function(ctx.space.population.positions)


def _model():
    space = SearchSpace(2, 1, [0.0], [1.0], device="cpu")
    space.population.initialize_static(torch.tensor([[0.4], [0.8]]))
    return Otorchmizer(space, HalvingOptimizer(), lambda x: x.square().sum(), save_agents=True)


@pytest.mark.parametrize(
    "budget,error",
    [(-1, ValueError), (np.int64(-1), ValueError), (1.5, TypeError), ("2", TypeError), (None, TypeError)],
)
def test_budget_validation_precedes_callbacks_evaluation_and_counter_changes(budget, error):
    model = _model()
    model.start(1)
    before = dill.dumps((model.iteration, model.total_iterations, model.n_iterations, model.history))
    positions = model.space.population.positions.clone()

    class RejectWork(Callback):
        def on_task_begin(self, model):
            pytest.fail("Invalid budgets must not dispatch callbacks.")

    with pytest.raises(error, match="n_iterations"):
        model.start(budget, [RejectWork()])

    assert before == dill.dumps((model.iteration, model.total_iterations, model.n_iterations, model.history))
    assert torch.equal(positions, model.space.population.positions)


def test_constructor_validates_history_and_callable_before_compilation():
    space = SearchSpace(2, 1, 0, 1, device="cpu")
    optimizer = HalvingOptimizer()
    with pytest.raises(TypeError, match="save_agents"):
        Otorchmizer(space, optimizer, lambda x: x.sum(), save_agents="yes")
    with pytest.raises(TypeError, match="pointer"):
        Otorchmizer(space, optimizer, None)
    assert not hasattr(optimizer, "compilations")


@pytest.mark.parametrize("params", [[], "", False, [("x", 1)]])
def test_optimizer_requires_mapping_even_for_falsey_parameters(params):
    with pytest.raises(TypeError, match="params"):
        Optimizer(params)
    optimizer = Optimizer()
    with pytest.raises(TypeError, match="params"):
        optimizer.build(params)
    assert optimizer.params == {}


def test_optimizer_accepts_read_only_mapping():
    assert Optimizer(MappingProxyType({"custom": 3})).custom == 3


@pytest.mark.parametrize(
    "adapter",
    [
        lambda: Function(lambda x: x.sum()),
        lambda: Function(lambda x: x.sum(dim=(1, 2)), batch=True),
        lambda: ConstrainedFunction(lambda x: x.sum(), [lambda x: x.sum() > 0]),
        lambda: MultiObjectiveWeightedFunction([lambda x: x.sum()], [1.0]),
    ],
)
def test_driver_preserves_existing_objective_adapters(adapter):
    function = adapter()
    model = Otorchmizer(SearchSpace(2, 1, 0, 1, device="cpu"), HalvingOptimizer(), function)
    assert model.function is function
    model.start(0)
    assert model.space.population.fitness.shape == (2,)


def test_raw_callable_and_model_replacement_are_adapted_before_update():
    model = _model()
    assert isinstance(model.function, Function)

    class Replace(Callback):
        def on_update_before(self, ctx):
            with pytest.raises(FrozenInstanceError):
                ctx.function = None
            ctx.space.population.positions.fill_(1.0)
            model.function = lambda x: 10 * x.sum()

    model.update([Replace()])
    torch.testing.assert_close(model.optimizer.seen, torch.full((2,), 5.0))
    model.evaluate()
    torch.testing.assert_close(model.space.population.fitness, torch.full((2,), 5.0))


def test_falsey_callback_sequences_preserve_order_and_are_not_reused():
    events = []

    class Falsey(list):
        def __bool__(self):
            return False

    class Recorder(Callback):
        def __init__(self, name):
            self.name = name

        def on_task_begin(self, model):
            events.append(("begin", self.name))

        def on_iteration_end(self, iteration, model):
            assert len(model.history.best_agent) == iteration
            events.append((iteration, self.name))

        def on_task_end(self, model):
            events.append(("end", self.name))

    model = _model()
    model.start(np.int64(2), Falsey([Recorder("a"), Recorder("b")]))
    assert events == [
        ("begin", "a"),
        ("begin", "b"),
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
        ("end", "a"),
        ("end", "b"),
    ]
    model.start(1)
    assert len(events) == 8
    assert model.total_iterations == 3
    assert model.iteration == 0
    assert model.optimizer.compilations == 1
    assert len(model.history.time) == 2


def test_driver_uses_monotonic_clock_and_is_quiet_by_default(monkeypatch, capsys):
    model = _model()
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("otorchmizer.otorchmizer.time.perf_counter", lambda: next(ticks))
    model.start(0)
    assert model.history.time == [2.5]
    assert capsys.readouterr() == ("", "")


def test_objective_failure_does_not_emit_task_end_or_elapsed_history():
    model = _model()
    ended = []

    class Recorder(Callback):
        def on_task_end(self, model):
            ended.append(True)

    def fail(position):
        raise RuntimeError("Objective failed.")

    model.function = fail
    with pytest.raises(RuntimeError, match="Objective failed"):
        model.start(1, [Recorder()])
    assert ended == []
    assert not hasattr(model.history, "time")


@pytest.mark.skipif(
    sys.platform == "win32" and torch.__version__.startswith("2.0."),
    reason="Torch 2.0 does not support torch.compile on Windows.",
)
def test_compiled_checkpoint_drops_dispatch_without_recompiling_buffers():
    model = _model()
    model.optimizer.buffer.fill_(17.0)
    model.optimizer.torch_compile(backend="eager")
    restored = dill.loads(dill.dumps(model))
    assert model.optimizer._compiled_update is not None
    assert restored.optimizer._compiled_update is None
    assert restored.optimizer.compilations == 1
    assert restored.optimizer.buffer.item() == 17.0
    restored.start(1)
    assert restored.optimizer.compilations == 1
    assert torch.isfinite(restored.space.best_fitness)


def test_existing_standard_logger_does_not_break_optimization():
    logger = logging.getLogger("otorchmizer.otorchmizer")
    assert type(logger) is logging.Logger
    _model().start(1)


def test_checkpoint_discards_unserializable_transient_dispatch_on_every_platform():
    model = _model()
    model.optimizer._compiled_update = (value for value in range(1))
    restored = dill.loads(dill.dumps(model))
    assert restored.optimizer._compiled_update is None
    assert restored.optimizer.compilations == 1
    restored.start(1)


def test_progress_display_requires_explicit_opt_in(capsys):
    _model().start(1, progress=True)
    assert "100%" in capsys.readouterr().err


def test_index_only_budget_is_normalized_before_callbacks_and_updates():
    seen = []

    class IndexOnly:
        def __index__(self):
            return 2

    class Step(Optimizer):
        def update(self, ctx):
            assert type(ctx.n_iterations) is int
            seen.append(ctx.n_iterations)
            ctx.space.population.positions.add_(1 / ctx.n_iterations)

    class Observe(Callback):
        def on_task_begin(self, model):
            assert type(model.n_iterations) is int
            seen.append(model.n_iterations)

    space = SearchSpace(2, 1, 0, 2, device="cpu")
    space.population.initialize_static(torch.zeros(2, 1))
    model = Otorchmizer(space, Step(), lambda x: x.sum())
    model.start(IndexOnly(), [Observe()])
    assert seen == [2, 2, 2]
    assert model.total_iterations == 2
    torch.testing.assert_close(space.population.positions, torch.ones(2, 1, 1))


@pytest.mark.parametrize("name", ["n_agents", "n_variables", "n_dimensions"])
@pytest.mark.parametrize("value,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_population_validates_counts(name, value, error):
    counts = {"n_agents": 2, "n_variables": 1, "n_dimensions": 1}
    counts[name] = value
    with pytest.raises(error, match=name):
        Population(**counts, lower_bound=torch.zeros(1), upper_bound=torch.ones(1))


@pytest.mark.parametrize("bounds", [([0, 2], [1, 3]), ([[[0]]], [1])])
def test_bounds_cannot_broadcast_into_extra_variables(bounds):
    with pytest.raises(ValueError, match="bound"):
        SearchSpace(2, 1, *bounds, device="cpu")


@pytest.mark.parametrize("bounds", [([2], [1]), ([float("nan")], [1]), ([0], [float("inf")])])
def test_numeric_spaces_reject_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bound"):
        SearchSpace(2, 1, *bounds, device="cpu")


@pytest.mark.parametrize(
    "mapping,error", [([], ValueError), (["a", "a"], ValueError), ("ab", TypeError), ([0, 1], TypeError)]
)
def test_population_mapping_has_one_unique_name_per_variable(mapping, error):
    with pytest.raises(error, match="mapping"):
        SearchSpace(2, 2, 0, 1, mapping=mapping, device="cpu")


@pytest.mark.parametrize("dtype,error", [(torch.long, ValueError), ("float64", TypeError)])
def test_population_rejects_invalid_dtype_before_transfer(dtype, error):
    with pytest.raises(error, match="dtype"):
        SearchSpace(2, 1, 0, 1, dtype=dtype, device="cpu")
    population = _model().space.population
    before = population.positions
    with pytest.raises(error, match="dtype"):
        population.to("cpu", dtype=dtype)
    assert population.positions is before


@pytest.mark.parametrize("method", ["initialize_uniform", "initialize_binary", "initialize_static"])
def test_initialization_resets_current_scores_and_complete_archive(method):
    population = _model().space.population
    population.fitness.zero_()
    population.update_best()
    values = torch.tensor([[0.9], [0.7]])
    if method == "initialize_static":
        population.initialize_static(values)
        values.zero_()
        assert population.positions[0].item() == pytest.approx(0.9)
    else:
        getattr(population, method)()
    assert torch.isposinf(population.fitness).all()
    assert torch.isposinf(population.best_fitness)
    assert torch.equal(population.best_position, population.positions[0])


def test_population_scatter_and_gather_own_their_shards_and_archives():
    population = _model().space.population
    original = population.positions.clone()
    original_best = population.best_position.clone()
    shards = population.scatter([torch.device("cpu"), torch.device("cpu")])
    shards[0].positions.fill_(9)
    shards[0].best_position.fill_(8)
    shards[0].best_fitness.fill_(1)
    assert torch.equal(population.positions, original)
    assert torch.equal(population.best_position, original_best)
    assert torch.equal(shards[1].best_position, original_best)
    gathered = Population.gather(shards, torch.device("cpu"))
    gathered.positions.zero_()
    gathered.best_position.zero_()
    gathered.best_fitness.zero_()
    assert shards[0].positions.item() == 9
    assert shards[0].best_position.item() == 8
    assert shards[0].best_fitness.item() == 1


def test_nan_fitness_is_rejected_instead_of_masking_other_agents():
    population = _model().space.population
    population.fitness = torch.tensor([torch.nan, 0.64])
    with pytest.raises(ValueError, match="fitness"):
        population.update_best()
    with pytest.raises(ValueError, match="fitness"):
        Function(lambda x: x[:, 0, 0], batch=True)(population.fitness.reshape(2, 1, 1))
    assert torch.isposinf(population.best_fitness)


def test_constructor_dtype_preserves_bounds_before_conversion():
    space = SearchSpace(2, 1, [1.00000001], [1.00000002], device="cpu", dtype=torch.float64)
    assert space.population.lb.item() == 1.00000001
    assert space.population.ub.item() == 1.00000002
    assert space.population.positions.dtype == torch.float64


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BooleanSpace(2, 1, device="cpu", dtype=torch.float64),
        lambda: HyperComplexSpace(2, 1, 3, device="cpu", dtype=torch.float64),
        lambda: ParetoSpace(torch.ones(2, 1), device="cpu", dtype=torch.float64),
        lambda: TreeSpace(2, 1, 0, 1, device="cpu", dtype=torch.float64),
    ],
)
def test_specialized_spaces_accept_constructor_dtype(factory):
    space = factory()
    assert space.population.positions.dtype == torch.float64
    if isinstance(space, TreeSpace):
        assert all(terminal.dtype == torch.float64 for terminal in space.terminals)
        assert space.best_tree.position.dtype == torch.float64
