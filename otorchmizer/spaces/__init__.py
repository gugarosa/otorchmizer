# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Spaces package for search space implementations."""

from otorchmizer.spaces.boolean import BooleanSpace
from otorchmizer.spaces.grid import GridSpace
from otorchmizer.spaces.hyper_complex import HyperComplexSpace
from otorchmizer.spaces.pareto import ParetoSpace
from otorchmizer.spaces.search import SearchSpace
from otorchmizer.spaces.tree import TreeSpace

__all__ = [
    "BooleanSpace",
    "GridSpace",
    "HyperComplexSpace",
    "ParetoSpace",
    "SearchSpace",
    "TreeSpace",
]
