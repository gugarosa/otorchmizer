# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from copy import copy

import torch

from otorchmizer.core import DeviceManager, Function, Population, Space
from otorchmizer.core.optimizer import UpdateContext
from otorchmizer.optimizers.swarm import PSO

# This example demonstrates distributing a large population across multiple GPUs
# Each sub-population runs independently, then results are merged

torch.manual_seed(0)


def sphere(x):
    """Compute sphere fitness over the variable and dimension axes.

    Args:
        x: Candidate or population position tensor.

    Returns:
        Squared norms over the final two axes.

    """

    return (x**2).sum(dim=(-1, -2))


# Check available GPUs
gpus = DeviceManager.available_gpus()
if len(gpus) < 2:
    print("Multi-GPU example requires 2+ GPUs. Using CPU simulation instead.")
    gpus = [torch.device("cpu"), torch.device("cpu")]

n_agents = 200
n_variables = 10
lower_bound = torch.tensor([-10.0] * n_variables)
upper_bound = torch.tensor([10.0] * n_variables)

space = Space(n_agents, n_variables, 1, lower_bound, upper_bound, device=gpus[0])
space.build()
pop = space.population

fn = Function(sphere)

# Evaluate initial fitness
pop.fitness = fn(pop.positions)
pop.update_best()
print(f"Initial best fitness: {pop.best_fitness.item():.4f}")

# Scatter population across devices
sub_pops = pop.scatter(gpus)
print(f"Split into {len(sub_pops)} sub-populations of {[s.n_agents for s in sub_pops]} agents")

# Run PSO independently on each sub-population
for i, (sub_pop, dev) in enumerate(zip(sub_pops, gpus)):
    opt = PSO()
    opt.compile(sub_pop)
    opt.evaluate(sub_pop, fn)

    # A shallow space copy reuses the initialized shard without another population allocation
    sub_space = copy(space)
    sub_space.population = sub_pop
    sub_space.device = dev

    for it in range(100):
        ctx = UpdateContext(sub_space, fn, it, 100, dev)
        opt(ctx)
        sub_pop.clip()
        opt.evaluate(sub_pop, fn)

    print(f"  GPU {i} ({dev}): best fitness = {sub_pop.best_fitness.item():.6e}")

# Gather back into a single population
merged = Population.gather(sub_pops, gpus[0])
print(f"\nMerged best fitness: {merged.best_fitness.item():.6e}")
print(f"Total agents: {merged.n_agents}")
