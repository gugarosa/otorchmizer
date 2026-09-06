# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Objective checkpoint regressions across batching modes."""

import dill
import pytest
import torch

from otorchmizer.core import Function
from otorchmizer.functions.constrained import ConstrainedFunction


def _scalar_objective(position):
    return position.square().sum()


def _batch_objective(positions):
    return positions.square().sum(dim=(1, 2))


def _manual_objective(position):
    return position.square().sum().item()


@pytest.mark.parametrize(
    ("pointer", "batch"),
    [(_scalar_objective, False), (_batch_objective, True), (_manual_objective, False)],
)
def test_function_checkpoint_preserves_evaluation(pointer, batch, caplog):
    positions = torch.tensor([[[1.0]], [[-2.0]]], dtype=torch.float64)
    function = Function(pointer, batch=batch)
    expected = function(positions)

    restored = dill.loads(dill.dumps(function))
    caplog.clear()
    actual = restored(positions)

    torch.testing.assert_close(actual, expected)
    assert restored.batch is batch
    assert not any("cannot use vmap" in record.message for record in caplog.records)


def test_constrained_checkpoint_rebuilds_objective_and_constraint_batchers():
    positions = torch.tensor([[[1.0]], [[-2.0]]], dtype=torch.float64)
    function = ConstrainedFunction(_scalar_objective, [lambda position: position.sum() > 0], penalty=2)
    function(positions)

    restored = dill.loads(dill.dumps(function))

    torch.testing.assert_close(restored(positions), positions.new_tensor([1.0, 12.0]))
    restored.constraints[:] = [lambda position: position.sum() < 0]
    torch.testing.assert_close(restored(positions), positions.new_tensor([3.0, 4.0]))
