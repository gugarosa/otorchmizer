"""Tests for math modules: random, distribution, general, hyper."""

import torch

from otorchmizer.math import distribution as d
from otorchmizer.math import general as g
from otorchmizer.math import hyper as h
from otorchmizer.math import random as r


class TestRandom:
    def test_binary(self):
        result = r.generate_binary_random_number(size=100)
        assert result.shape == (100,)
        assert all(v in [0.0, 1.0] for v in result.unique().tolist())

    def test_uniform(self):
        result = r.generate_uniform_random_number(low=-5.0, high=5.0, size=100)
        assert result.shape == (100,)
        assert result.min() >= -5.0
        assert result.max() <= 5.0

    def test_gaussian(self):
        result = r.generate_gaussian_random_number(mean=0.0, variance=1.0, size=1000)
        assert result.shape == (1000,)
        assert abs(result.mean().item()) < 0.2  # should be near 0

    def test_integer(self):
        result = r.generate_integer_random_number(low=0, high=10, size=50)
        assert result.shape == (50,)
        assert result.min() >= 0
        assert result.max() < 10

    def test_integer_scalar(self):
        result = r.generate_integer_random_number(low=0, high=5)
        assert isinstance(result, int)
        assert 0 <= result < 5

    def test_exponential(self):
        result = r.generate_exponential_random_number(scale=1.0, size=100)
        assert result.shape == (100,)
        assert (result >= 0).all()


class TestDistribution:
    def test_bernoulli(self):
        result = d.generate_bernoulli_distribution(prob=0.5, size=100)
        assert result.shape == (100,)

    def test_levy(self):
        result = d.generate_levy_distribution(beta=1.5, size=50)
        assert result.shape == (50,)

    def test_choice(self):
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = d.generate_choice_distribution(n=4, probs=probs, size=2)
        assert result.shape == (2,)


class TestGeneral:
    def test_euclidean_distance(self):
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 0.0, 0.0])
        dist = g.euclidean_distance(x, y)
        assert abs(dist.item() - 1.0) < 1e-6

    def test_pairwise_distances(self):
        positions = torch.rand(10, 3, 1)
        dists = g.pairwise_distances(positions)
        assert dists.shape == (10, 10)
        assert (dists.diagonal() < 1e-6).all()  # self-distance is 0

    def test_tournament_selection(self):
        fitness = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
        selected = g.tournament_selection(fitness, n=3, size=2)
        assert selected.shape == (3,)
        assert (selected < 5).all()

    def test_n_wise(self):
        items = [1, 2, 3, 4, 5, 6]
        pairs = list(g.n_wise(items, 2))
        assert pairs == [(1, 2), (3, 4), (5, 6)]

    def test_kmeans(self):
        x = torch.randn(50, 3, 1)
        labels = g.kmeans_torch(x, n_clusters=3)
        assert labels.shape == (50,)
        assert labels.unique().numel() <= 3


class TestHyper:
    def test_norm(self):
        array = torch.ones(5, 4)
        result = h.norm(array)
        assert result.shape == (5,)
        assert abs(result[0].item() - 2.0) < 1e-6  # sqrt(4)

    def test_span(self):
        array = torch.ones(3, 4)
        lb = torch.tensor([0.0, 0.0, 0.0])
        ub = torch.tensor([10.0, 10.0, 10.0])
        result = h.span(array, lb, ub)
        assert result.shape == (3,)
