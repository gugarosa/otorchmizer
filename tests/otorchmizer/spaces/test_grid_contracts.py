# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from otorchmizer.spaces.grid import GridSpace


def test_grid_space_does_not_overshoot_upper_bounds():
    space = GridSpace(1, step=0.6, lower_bound=0.0, upper_bound=1.0, device="cpu")

    torch.testing.assert_close(space.population.positions[:, 0, 0], torch.tensor([0.0, 0.6]))
    assert (space.population.positions <= space.population.ub).all()


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan"), float("inf")])
def test_grid_space_rejects_invalid_steps(step):
    with pytest.raises(ValueError, match="step"):
        GridSpace(1, step=step, lower_bound=0.0, upper_bound=1.0, device="cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "step,lower,upper,expected",
    [
        (0.1, 0.0, 0.9, [i / 10 for i in range(10)]),
        (0.1, -0.3, 0.0, [-0.3, -0.2, -0.1, 0]),
        (0.2, -0.3, 0.25, [-0.3, -0.1, 0.1]),
        (0.6, 0.0, 1.0, [0.0, 0.6]),
        (3.0, 1.0, 1.0, [1.0]),
    ],
)
def test_grid_retains_rounded_endpoints_but_never_a_full_extra_step(dtype, step, lower, upper, expected):
    space = GridSpace(1, step, lower, upper, device="cpu", dtype=dtype)
    torch.testing.assert_close(space.grid[:, 0], torch.tensor(expected, dtype=dtype))
    assert (space.population.positions <= space.population.ub).all()
    assert (space.population.positions >= space.population.lb).all()
    assert space.population.positions.dtype == dtype


def test_grid_keeps_precise_bounds_and_steps_until_requested_conversion():
    space = GridSpace(1, 1e-8, 1.00000001, 1.00000002, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(
        space.grid[:, 0], torch.tensor([1.00000001, 1.00000002], dtype=torch.float64), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
def test_half_grid_uses_wide_intermediates_without_losing_representable_points(device):
    space = GridSpace(1, 16384, -49152, 49152, dtype=torch.float16, device=device)
    expected = torch.tensor([-49152, -32768, -16384, 0, 16384, 32768, 49152], dtype=torch.float16, device=device)
    torch.testing.assert_close(space.grid[:, 0], expected, rtol=0, atol=0)
    assert space.n_agents == 7
    assert torch.isfinite(space.grid).all()
    assert (space.grid[:, 0] >= space.population.lb[0, 0]).all()
    assert (space.grid[:, 0] <= space.population.ub[0, 0]).all()
