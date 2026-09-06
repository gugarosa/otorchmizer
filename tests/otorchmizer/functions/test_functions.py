# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tests for functions — constrained and multi-objective wrappers."""

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.functions.constrained import ConstrainedFunction
from otorchmizer.functions.multi_objective.standard import MultiObjectiveFunction
from otorchmizer.functions.multi_objective.weighted import MultiObjectiveWeightedFunction


class TestConstrainedFunction:
    def test_no_violation(self):
        fn = ConstrainedFunction(
            pointer=lambda x: (x**2).sum(),
            constraints=[lambda x: torch.tensor(True)],
            penalty=100.0,
        )
        positions = torch.rand(5, 2, 1)
        result = fn(positions)
        assert result.shape == (5,)

    def test_penalty_applied_on_violation(self):
        def base_fn(x):
            return (x**2).sum()

        satisfied_fn = ConstrainedFunction(
            pointer=base_fn,
            constraints=[lambda x: torch.tensor(True)],
            penalty=10.0,
        )
        violated_fn = ConstrainedFunction(
            pointer=base_fn,
            constraints=[lambda x: torch.tensor(False)],
            penalty=10.0,
        )
        positions = torch.ones(3, 2, 1)
        fit_ok = satisfied_fn(positions)
        fit_bad = violated_fn(positions)
        # violated fitness should be larger
        assert (fit_bad >= fit_ok).all()

    def test_batch_constraint(self):
        def batch_constraint(positions):
            return positions[:, 0, 0] > 0.5

        fn = ConstrainedFunction(
            pointer=lambda x: (x**2).sum(dim=(1, 2)),
            constraints=[batch_constraint],
            penalty=5.0,
            batch=True,
        )
        positions = torch.rand(10, 2, 1)
        result = fn(positions)
        assert result.shape == (10,)
        base = (positions**2).sum(dim=(1, 2))
        expected = base + (positions[:, 0, 0] <= 0.5).to(base.dtype) * 5.0 * base
        torch.testing.assert_close(result, expected)

    def test_zero_penalty(self):
        fn = ConstrainedFunction(
            pointer=lambda x: (x**2).sum(),
            constraints=[lambda x: torch.tensor(False)],
            penalty=0.0,
        )
        positions = torch.rand(3, 2, 1)
        result = fn(positions)
        assert result.shape == (3,)

    def test_invalid_constraints_type(self):
        with pytest.raises(e.TypeError):
            ConstrainedFunction(
                pointer=lambda x: x.sum(),
                constraints="not_a_list",
                penalty=1.0,
            )

    def test_invalid_penalty_type(self):
        with pytest.raises(e.TypeError):
            ConstrainedFunction(
                pointer=lambda x: x.sum(),
                constraints=[],
                penalty="bad",
            )

    def test_negative_penalty(self):
        with pytest.raises(e.ValueError):
            ConstrainedFunction(
                pointer=lambda x: x.sum(),
                constraints=[],
                penalty=-1.0,
            )

    def test_multiple_constraints(self):
        fn = ConstrainedFunction(
            pointer=lambda x: (x**2).sum(),
            constraints=[
                lambda x: torch.tensor(True),
                lambda x: torch.tensor(True),
            ],
            penalty=1.0,
        )
        positions = torch.rand(3, 2, 1)
        result = fn(positions)
        assert result.shape == (3,)


class TestMultiObjectiveFunction:
    def test_two_objectives(self):
        fn = MultiObjectiveFunction(
            functions=[
                lambda x: (x**2).sum(),
                lambda x: (x**3).sum(),
            ]
        )
        positions = torch.rand(5, 2, 1)
        result = fn(positions)
        assert result.shape == (5, 2)

    def test_single_objective(self):
        fn = MultiObjectiveFunction(functions=[lambda x: (x**2).sum()])
        positions = torch.rand(3, 2, 1)
        result = fn(positions)
        assert result.shape == (3, 1)

    def test_invalid_functions_type(self):
        with pytest.raises(e.TypeError):
            MultiObjectiveFunction(functions="not_a_list")


class TestMultiObjectiveWeightedFunction:
    def test_weighted_scalarization(self):
        fn = MultiObjectiveWeightedFunction(
            functions=[
                lambda x: (x**2).sum(),
                lambda x: (x * 0).sum(),
            ],
            weights=[1.0, 0.0],
        )
        positions = torch.rand(4, 2, 1)
        result = fn(positions)
        assert result.shape == (4,)

    def test_equal_weights(self):
        fn = MultiObjectiveWeightedFunction(
            functions=[
                lambda x: (x * 0 + 2.0).sum(),
                lambda x: (x * 0 + 4.0).sum(),
            ],
            weights=[0.5, 0.5],
        )
        positions = torch.rand(3, 1, 1)
        result = fn(positions)
        assert result.shape == (3,)
        # each agent should get 0.5*2+0.5*4=3.0
        assert torch.allclose(result, torch.tensor([3.0, 3.0, 3.0]))

    def test_invalid_weights_type(self):
        with pytest.raises(e.TypeError):
            MultiObjectiveWeightedFunction(
                functions=[lambda x: x.sum()],
                weights="not_a_list",
            )

    def test_weights_size_mismatch(self):
        with pytest.raises(e.SizeError):
            MultiObjectiveWeightedFunction(
                functions=[lambda x: x.sum(), lambda x: x.sum()],
                weights=[1.0],
            )
