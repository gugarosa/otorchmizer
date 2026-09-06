# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import sys

import dill
import pytest
import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Optimizer
from otorchmizer.functions import ConstrainedFunction
from otorchmizer.functions.multi_objective import MultiObjectiveWeightedFunction
from otorchmizer.spaces import SearchSpace


def test_batch_configuration_changes_and_checkpoints_use_the_same_dispatch():
    calls = []

    def objective(positions):
        calls.append(positions.ndim)
        return positions.sum() if positions.ndim == 2 else positions.sum(dim=(1, 2))

    function = Function(objective)
    values = torch.ones(2, 3, 1)
    torch.testing.assert_close(function(values), torch.full((2,), 3.0))
    function.batch = True
    torch.testing.assert_close(function(values), torch.full((2,), 3.0))
    assert calls == [2, 3]
    restored = dill.loads(dill.dumps(function))
    torch.testing.assert_close(restored(values), torch.full((2,), 3.0))
    assert restored.batch is True
    function.batch = False
    assert function._manual is False
    torch.testing.assert_close(function(values), torch.full((2,), 3.0))
    assert calls[-1] == 2


def test_batch_change_resets_cached_manual_fallback():
    function = Function(lambda position: position.sum().item())
    function(torch.ones(2, 1, 1))
    assert function._manual
    function.batch = True
    assert not function._manual
    with pytest.raises(TypeError, match="fitness"):
        function(torch.ones(2, 1, 1))


@pytest.mark.parametrize("batch", ["yes", 1, None])
def test_batch_requires_boolean(batch):
    with pytest.raises(TypeError, match="batch"):
        Function(lambda x: x.sum(), batch=batch)


@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_satisfied_constraints_preserve_infinite_objective(penalty):
    function = ConstrainedFunction(lambda x: x.sum(), [lambda x: True], penalty=penalty)
    result = function(torch.tensor([[[torch.inf]], [[1.0]]]))
    assert torch.isposinf(result[0])
    assert result[1].item() == 1


def test_zero_penalty_preserves_violating_infinite_objective():
    function = ConstrainedFunction(lambda x: x.sum(), [lambda x: False], penalty=0)
    assert torch.isposinf(function(torch.full((1, 1, 1), torch.inf))).all()


def test_weighted_objective_preserves_double_precision_weights_and_mutations():
    weights = [1.00000001]
    function = MultiObjectiveWeightedFunction([lambda x: x.sum()], weights)
    values = torch.ones(2, 1, 1, dtype=torch.float64)
    assert function(values)[0].item() == 1.00000001
    weights[0] = 1.00000002
    assert function(values)[0].item() == 1.00000002
    weights.append(1.0)
    with pytest.raises(ValueError, match="weights"):
        function(values)


def test_constrained_constructor_rejects_noncallable_constraints():
    with pytest.raises(TypeError, match="constraints"):
        ConstrainedFunction(lambda x: x.sum(), [1])


def test_weighted_integer_objectives_do_not_truncate_fractional_weights():
    function = MultiObjectiveWeightedFunction([lambda x: x.new_tensor(2, dtype=torch.long)], [0.125])
    torch.testing.assert_close(
        function(torch.ones(2, 1, 1, dtype=torch.float64)), torch.full((2,), 0.25, dtype=torch.float64)
    )


def test_weighted_objective_rejects_non_vector_weights():
    function = MultiObjectiveWeightedFunction([lambda x: x.sum()], [[1.0]])
    with pytest.raises(ValueError, match="weights"):
        function(torch.ones(2, 1, 1))


def _compiled_objective(kind):
    if kind == "constrained":
        return ConstrainedFunction(lambda x: x.sum(dim=(1, 2)), [lambda x: x[:, 0, 0] > 0], penalty=1, batch=True)
    if kind == "weighted":
        return MultiObjectiveWeightedFunction([lambda x: x.sum(dim=(1, 2))], [0.0], batch=True)
    return Function(lambda x: x.sum(dim=(1, 2)), batch=True)


@pytest.mark.parametrize("kind,invalid", [("native", torch.nan), ("constrained", -torch.inf), ("weighted", torch.inf)])
def test_adapters_reject_nan_generated_by_their_final_transformation(kind, invalid):
    with pytest.raises(ValueError, match="fitness"):
        _compiled_objective(kind)(torch.full((2, 1, 1), invalid))


@pytest.mark.skipif(
    sys.platform == "win32" and torch.__version__.startswith("2.0."),
    reason="Torch 2.0 does not support torch.compile on Windows.",
)
@pytest.mark.parametrize("kind,invalid", [("native", torch.nan), ("constrained", -torch.inf), ("weighted", torch.inf)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_fullgraph_updates_validate_nan_without_invalidating_cuda_context(kind, invalid, device):
    class Evaluate(Optimizer):
        def update(self, ctx):
            ctx.space.population.fitness = ctx.function(ctx.space.population.positions)

    space = SearchSpace(2, 1, 0, 2, device=device)
    space.population.positions.fill_(1)
    optimizer = Evaluate()
    model = Otorchmizer(space, optimizer, _compiled_objective(kind))
    optimizer.torch_compile(backend="eager", fullgraph=True)
    model.update()
    expected = torch.zeros(2, device=device) if kind == "weighted" else torch.ones(2, device=device)
    torch.testing.assert_close(space.population.fitness, expected)
    space.population.positions.fill_(invalid)
    with pytest.raises(RuntimeError, match="fitness.*NaN"):
        model.update()
    space.population.positions.fill_(1)
    model.update()
    torch.testing.assert_close(space.population.fitness, expected)
