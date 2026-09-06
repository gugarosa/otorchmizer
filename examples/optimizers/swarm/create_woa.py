# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from otorchmizer.optimizers.swarm import WOA

# Creates a Whale Optimization Algorithm optimizer
# with default parameters (a_min_constant=2)
o = WOA()

# Prints out some properties
print(f"Algorithm: {o.algorithm}")
print(f"Built: {o.built}")
