# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Device-local distribution sampling regressions."""

from math import sqrt

import pytest
import torch

from otorchmizer.math.distribution import generate_levy_distribution
from otorchmizer.math.random import generate_exponential_random_number, generate_gamma_random_number


@pytest.mark.parametrize("sample", [generate_exponential_random_number, generate_gamma_random_number])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distribution_sampling_preserves_requested_dtype(sample, dtype):
    values = sample(scale=2.0, size=(8, 3), dtype=dtype)

    assert values.dtype is dtype
    assert values.device.type == "cpu"
    assert values.shape == (8, 3)
    assert torch.isfinite(values).all()
    assert (values >= 0).all()


@pytest.mark.parametrize(
    "sample", [generate_exponential_random_number, generate_gamma_random_number, generate_levy_distribution]
)
def test_distribution_sampling_preserves_the_global_dtype_default(sample):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        values = sample(size=8)
    finally:
        torch.set_default_dtype(previous)

    assert values.dtype is torch.float64


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "sample", [generate_exponential_random_number, generate_gamma_random_number, generate_levy_distribution]
)
def test_cuda_distribution_sampling_uses_the_cuda_generator(sample):
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()

    values = sample(size=128, device=torch.device("cuda:0"), dtype=torch.float64)

    assert values.is_cuda
    assert values.dtype is torch.float64
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert not torch.equal(cuda_state, torch.cuda.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_levy_sampling_preserves_requested_dtype(dtype):
    values = generate_levy_distribution(beta=1.5, size=(8, 3), dtype=dtype)

    assert values.dtype is dtype
    assert values.shape == (8, 3)
    assert torch.isfinite(values).all()


def test_levy_gaussian_endpoint_has_nonzero_normal_scale(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda size, **kwargs: torch.ones(size, **kwargs))

    values = generate_levy_distribution(beta=2, size=8, dtype=torch.float64)

    torch.testing.assert_close(values, torch.full((8,), sqrt(2), dtype=torch.float64))


@pytest.mark.parametrize("beta", [0, -1, 2.1, torch.nan])
def test_levy_rejects_invalid_stability_exponents(beta):
    with pytest.raises(ValueError, match="`beta`"):
        generate_levy_distribution(beta=beta)
