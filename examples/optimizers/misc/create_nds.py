# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Rank precomputed minimization objectives without scalarizing them."""

import torch

from otorchmizer.optimizers.misc import NDS
from otorchmizer.spaces import ParetoSpace

points = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
space = ParetoSpace(points, device="auto")
optimizer = NDS({"maximize": False})
optimizer.compile(space.population)
optimizer.evaluate(space.population)

print(f"Front ranks: {optimizer.status.tolist()}")
print(f"Nondominated points: {space.population.positions[optimizer.status == 0].squeeze(-1)}")
