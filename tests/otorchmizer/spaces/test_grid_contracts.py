# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import otorchmizer.utils.exception as e
from otorchmizer.spaces.grid import GridSpace


def test_grid_space_does_not_overshoot_upper_bounds():
    space = GridSpace(1, step=0.6, lower_bound=0.0, upper_bound=1.0, device="cpu")

    torch.testing.assert_close(space.population.positions[:, 0, 0], torch.tensor([0.0, 0.6]))
    assert (space.population.positions <= space.population.ub).all()


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan"), float("inf")])
def test_grid_space_rejects_invalid_steps(step):
    with pytest.raises(e.ValueError, match="step"):
        GridSpace(1, step=step, lower_bound=0.0, upper_bound=1.0, device="cpu")
