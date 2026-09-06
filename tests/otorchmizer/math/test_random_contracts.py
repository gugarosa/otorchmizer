# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from otorchmizer.math.random import generate_integer_random_number


@pytest.mark.parametrize("exclude", [0, 2, 4])
def test_integer_sampling_excludes_values_without_retrying(exclude):
    torch.manual_seed(17)

    values = generate_integer_random_number(0, 5, exclude_value=exclude, size=(1024,))

    assert values.dtype == torch.int64
    assert set(values.tolist()) == set(range(5)) - {exclude}


def test_integer_sampling_rejects_an_empty_allowed_range():
    with pytest.raises(ValueError, match="exclude_value"):
        generate_integer_random_number(0, 1, exclude_value=0)


def test_integer_sampling_preserves_empty_output():
    values = generate_integer_random_number(0, 1, exclude_value=0, size=(0, 3))

    assert values.shape == (0, 3)
    assert values.dtype == torch.int64


def test_integer_sampling_scalar_returns_python_integer():
    value = generate_integer_random_number(3, 5, exclude_value=3)

    assert type(value) is int
    assert value == 4
