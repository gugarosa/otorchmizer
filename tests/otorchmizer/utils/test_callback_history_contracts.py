# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Optimizer
from otorchmizer.spaces import HyperComplexSpace, SearchSpace
from otorchmizer.utils.callback import Callback, CheckpointCallback, DiscreteSearchCallback
from otorchmizer.utils.history import History


@pytest.mark.parametrize("path", [Path("checkpoints") / "model.pkl", Path.cwd() / "model.pkl"])
@pytest.mark.parametrize("path_type", [str, Path])
def test_checkpoint_prefix_preserves_directory(path, path_type):
    saved = []
    callback = CheckpointCallback(path_type(path), frequency=2)
    callback.on_iteration_end(1, SimpleNamespace(save=saved.append))
    callback.on_iteration_end(2, SimpleNamespace(save=saved.append))
    assert saved == [path.with_name("iter_2_model.pkl")]


@pytest.mark.parametrize(
    "path,frequency,error",
    [
        (1, 1, TypeError),
        ("", 1, ValueError),
        ("model.pkl", -1, ValueError),
        ("model.pkl", 1.5, TypeError),
        ("model.pkl", True, TypeError),
    ],
)
def test_checkpoint_validates_path_and_frequency(path, frequency, error):
    with pytest.raises(error):
        CheckpointCallback(path, frequency)


def test_checkpoint_roundtrip_uses_real_path_without_retaining_callbacks():
    path = Path(__file__).with_name(f"checkpoint-{uuid4().hex}.pkl")
    saved_path = path.with_name(f"iter_1_{path.name}")

    class Stationary(Optimizer):
        def update(self, ctx):
            pass

    model = Otorchmizer(SearchSpace(2, 1, 0, 1, device="cpu"), Stationary(), lambda x: x.sum())
    try:
        model.start(1, [CheckpointCallback(path, 1)])
        restored = Otorchmizer.load(saved_path)
        assert restored.total_iterations == 1
        assert not hasattr(restored.history, "time")
        restored.start(1)
        assert restored.total_iterations == 2
        assert not path.with_name(f"iter_2_{path.name}").exists()
        assert torch.equal(restored.space.best_position, model.space.best_position)
    finally:
        saved_path.unlink(missing_ok=True)


@pytest.mark.parametrize("values", [[], 0, [[0, 1]], [float("nan")], [float("inf")], [-1], [2]])
def test_discrete_search_rejects_invalid_vectors_before_mutating_positions(values):
    space = SearchSpace(2, 1, 0, 1, device="cpu")
    before = space.population.positions.clone()
    callback = DiscreteSearchCallback([values])
    with pytest.raises(ValueError, match="allowed_values"):
        callback.on_task_begin(SimpleNamespace(space=space))
    assert torch.equal(space.population.positions, before)


def test_discrete_search_revalidates_all_mutated_vectors_before_any_projection():
    space = HyperComplexSpace(1, 2, 2, device="cpu")
    space.population.positions.fill_(0.2)
    values = [[0, 1], [0, 1]]
    callback = DiscreteSearchCallback(values)
    callback.on_task_begin(SimpleNamespace(space=space))
    values[1] = [2]
    with pytest.raises(ValueError, match="allowed_values"):
        callback.on_evaluate_before(space.population, None)
    assert (space.population.positions == 0.2).all()


def test_discrete_search_projects_dimensions_with_first_value_tie_break():
    space = HyperComplexSpace(1, 1, 3, device="cpu", dtype=torch.float64)
    space.population.positions[:] = torch.tensor([[[0.2, 0.8, 0.5]]])
    DiscreteSearchCallback([[0, 1]]).on_evaluate_before(space.population, None)
    assert space.population.positions.tolist() == [[[0.0, 1.0, 0.0]]]


def test_history_selects_agent_axis_without_changing_native_shapes_or_snapshots():
    history = History(save_agents=True)
    positions = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    fitness = torch.tensor([1.0, 2.0])
    for _ in range(3):
        history.dump(positions=positions, fitness=fitness, best_agent=(positions[0], fitness[0]))
    positions.zero_()
    fitness.zero_()
    assert history.get_convergence("positions").shape == (3, 2, 2, 3)
    assert history.get_convergence("fitness").shape == (3, 2)
    assert history.get_convergence("positions", index=1).shape == (3, 2, 3)
    np.testing.assert_array_equal(history.get_convergence("fitness", index=np.int64(1)), [2, 2, 2])
    assert history.get_convergence("positions", index=1)[0, 0, 0] == 6
    best_positions, best_fitness = history.get_convergence("best_agent")
    assert best_positions.shape == (3, 2, 3)
    assert best_fitness.shape == (3,)
    with pytest.raises(ValueError, match="index"):
        history.get_convergence("best_agent", index=0)
    with pytest.raises(TypeError, match="index"):
        history.get_convergence("positions", index=0.5)
    with pytest.raises(IndexError):
        history.get_convergence("fitness", index=2)


def test_iteration_callbacks_see_clipping_then_evaluation_then_history():
    events = []

    class Move(Optimizer):
        def update(self, ctx):
            ctx.space.population.positions.fill_(2)
            events.append("update")

    class Observe(Callback):
        def on_task_begin(self, model):
            events.append("task_begin")

        def on_evaluate_before(self, population, function):
            events.append("evaluate_before")

        def on_evaluate_after(self, population, function):
            assert (population.positions <= 1).all()
            events.append("evaluate_after")

        def on_iteration_begin(self, iteration, model):
            events.append("iteration_begin")

        def on_update_before(self, ctx):
            events.append("update_before")

        def on_update_after(self, ctx):
            assert (ctx.space.population.positions == 2).all()
            events.append("update_after")

        def on_iteration_end(self, iteration, model):
            assert len(model.history.best_agent) == 1
            events.append("iteration_end")

        def on_task_end(self, model):
            assert not hasattr(model.history, "time")
            events.append("task_end")

    model = Otorchmizer(SearchSpace(2, 1, 0, 1, device="cpu"), Move(), lambda x: x.sum())
    model.start(1, [Observe()])
    assert events == [
        "task_begin",
        "evaluate_before",
        "evaluate_after",
        "iteration_begin",
        "update_before",
        "update",
        "update_after",
        "evaluate_before",
        "evaluate_after",
        "iteration_end",
        "task_end",
    ]


def test_history_selects_each_record_before_stacking_a_shrinking_population():
    history = History(save_agents=True)
    for values in ([1.0, 2.0, 3.0], [4.0, 5.0]):
        fitness = torch.tensor(values)
        history.dump(positions=fitness.reshape(-1, 1, 1), fitness=fitness)

    np.testing.assert_array_equal(history.get_convergence("positions", index=0), [[[1.0]], [[4.0]]])
    np.testing.assert_array_equal(history.get_convergence("fitness", index=0), [1.0, 4.0])
    for key in ("positions", "fitness"):
        with pytest.raises(IndexError):
            history.get_convergence(key, index=2)
        records = history.get_convergence(key)
        assert records.shape == (2,)
        assert records.dtype == object
        assert records[0].shape[0] == 3
        assert records[1].shape[0] == 2
        records[0].fill(99)
        assert history.get_convergence(key, index=0).reshape(-1)[0] == 1
