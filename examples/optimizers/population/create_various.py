# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from otorchmizer.optimizers.population import GWO, HHO

# Grey Wolf Optimizer
gwo = GWO()
print(f"Algorithm: {gwo.algorithm}")

# Harris Hawks Optimization
hho = HHO()
print(f"Algorithm: {hho.algorithm}")
