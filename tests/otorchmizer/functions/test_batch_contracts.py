# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from otorchmizer.core.function import Function
from otorchmizer.functions.constrained import ConstrainedFunction
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction


def _branched_scalar(position):
    total = position.sum()
    return total if total.item() >= 0 else -total


def test_function_evaluates_vmap_incompatible_scalar_objectives():
    function = Function(_branched_scalar)
    positions = torch.tensor([[[-2.0]], [[3.0]]], dtype=torch.float64)

    torch.testing.assert_close(function(positions), torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_function_manual_evaluation_preserves_python_scalar_precision():
    function = Function(lambda position: position.sum().item())
    positions = torch.tensor([[[1.00000004]], [[1.00000001]]], dtype=torch.float64)

    result = function(positions)

    assert result.dtype == torch.float64
    torch.testing.assert_close(result, positions[:, 0, 0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "message",
    [
        "vmap: this test operation has no batching support",
        "DispatchKey FuncTorchBatched doesn't correspond to a device",
    ],
)
def test_function_resolves_unsupported_vmap_once(monkeypatch, message):
    calls = []
    warnings = []

    def unsupported_vmap(pointer):
        def mapped(positions):
            calls.append("vmap")
            raise RuntimeError(message)

        return mapped

    monkeypatch.setattr(torch, "vmap", unsupported_vmap)
    monkeypatch.setattr("otorchmizer.core.function.logger.warning", warnings.append)
    function = Function(lambda position: position.sum())
    positions = torch.ones(2, 1, 1)

    assert torch.equal(function(positions), torch.ones(2))
    assert torch.equal(function(positions), torch.ones(2))
    assert calls == ["vmap"]
    assert len(warnings) == 1
    assert "pointer=" in warnings[0]


def test_function_does_not_retry_user_runtime_errors():
    calls = []
    failure = RuntimeError("objective failed")

    def objective(position):
        calls.append("called")
        raise failure

    with pytest.raises(RuntimeError) as error:
        Function(objective)(torch.ones(2, 1, 1))

    assert error.value is failure
    assert calls == ["called"]


@pytest.mark.parametrize("batch", [False, True])
def test_function_rejects_non_scalar_fitness_per_agent(batch):
    function = Function(lambda positions: positions, batch=batch)

    with pytest.raises(ValueError, match="fitness"):
        function(torch.ones(2, 3, 1))


def test_constrained_function_supports_scalar_boolean_predicates():
    function = ConstrainedFunction(
        _branched_scalar,
        [lambda position: position.sum().item() >= 0],
        penalty=2.0,
    )
    positions = torch.tensor([[[-2.0]], [[3.0]]], dtype=torch.float64)

    result = function(positions)

    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.tensor([6.0, 3.0], dtype=torch.float64))


def test_constrained_function_tracks_mutated_constraints():
    function = ConstrainedFunction(lambda position: position.sum(), [], penalty=2.0)
    positions = torch.ones(2, 1, 1)
    assert torch.equal(function(positions), torch.ones(2))

    function.constraints.append(lambda position: position.new_tensor(False, dtype=torch.bool))
    assert torch.equal(function(positions), torch.full((2,), 3.0))

    function.constraints[0] = lambda position: position.new_tensor(True, dtype=torch.bool)
    assert torch.equal(function(positions), torch.ones(2))


def test_constrained_function_preserves_float64_penalty():
    function = ConstrainedFunction(
        lambda position: position.sum(),
        [lambda position: position.new_tensor(False, dtype=torch.bool)],
        penalty=1.0 + 2**-30,
    )
    result = function(torch.ones(2, 1, 1, dtype=torch.float64))

    torch.testing.assert_close(result, torch.full((2,), 2.0 + 2**-30, dtype=torch.float64), rtol=0, atol=0)


def test_multi_objective_function_reuses_scalar_batching_contract():
    function = MultiObjectiveFunction([_branched_scalar, lambda position: position.sum()])
    positions = torch.tensor([[[-2.0]], [[3.0]]])

    torch.testing.assert_close(function(positions), torch.tensor([[2.0, -2.0], [3.0, 3.0]]))
