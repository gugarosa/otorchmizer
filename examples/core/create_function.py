import torch

from otorchmizer.core import Function


# Defines a function with a tensor input and a scalar return
def sphere(x):
    return (x ** 2).sum()


# Any type of PyTorch-compatible function can be used as a pointer.
# The Function class auto-vectorizes it across the population via torch.vmap.
f = Function(sphere)

# Evaluates across a batch of agents
positions = torch.rand(10, 3, 1)  # 10 agents, 3 variables, 1 dimension
fitness = f(positions)

# Prints out some properties
print(f"Function: {f.name}")
print(f"Input shape: {positions.shape}")
print(f"Output shape: {fitness.shape}")
print(f"Fitness values: {fitness}")
