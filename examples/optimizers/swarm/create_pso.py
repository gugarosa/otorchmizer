# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from otorchmizer.optimizers.swarm import PSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"w": 0.7, "c1": 1.7, "c2": 1.7}

# Creates a PSO optimizer
o = PSO(params=params)

# Prints out some properties
print(f"Algorithm: {o.algorithm}")
print(f"Parameters: {o.params}")
print(f"Built: {o.built}")
