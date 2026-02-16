import torch

from otorchmizer.core import DeviceManager, Function, Population, Space
from otorchmizer.optimizers.swarm import PSO
from otorchmizer.core.optimizer import UpdateContext

# This example demonstrates distributing a large population across multiple GPUs.
# Each sub-population runs independently, then results are merged.

torch.manual_seed(0)


def sphere(x):
    return (x ** 2).sum(dim=(-1, -2))


# Check available GPUs
gpus = DeviceManager.available_gpus()
if len(gpus) < 2:
    print("Multi-GPU example requires 2+ GPUs. Using CPU simulation instead.")
    gpus = [torch.device("cpu"), torch.device("cpu")]

# Create a large population on the first device
n_agents = 200
n_variables = 10
lower_bound = torch.tensor([-10.0] * n_variables)
upper_bound = torch.tensor([10.0] * n_variables)

pop = Population(n_agents, n_variables, 1, lower_bound, upper_bound, device=gpus[0])
pop.initialize_uniform()

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

    space = Space.__new__(Space)
    space.population = sub_pop

    for it in range(100):
        ctx = UpdateContext(space, fn, it, 100, dev)
        opt.update(ctx)
        sub_pop.clip()
        opt.evaluate(sub_pop, fn)

    print(f"  GPU {i} ({dev}): best fitness = {sub_pop.best_fitness.item():.6e}")

# Gather back into a single population
merged = Population.gather(sub_pops, gpus[0])
print(f"\nMerged best fitness: {merged.best_fitness.item():.6e}")
print(f"Total agents: {merged.n_agents}")
