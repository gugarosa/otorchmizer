import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.swarm import PSO

# Random seed for experimental consistency
torch.manual_seed(0)


def sphere(x):
    return (x ** 2).sum(dim=(-1, -2))


# Number of agents and decision variables
n_agents = 50
n_variables = 5

# Lower and upper bounds
lower_bound = [-10] * n_variables
upper_bound = [10] * n_variables

# Creates the space on GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound, device=device)
space.build()
optimizer = PSO()
function = Function(sphere)

# Enables torch.compile for JIT acceleration
optimizer.compile(space.population)
optimizer.torch_compile(mode="reduce-overhead")

# Bundles every piece into Otorchmizer class
opt = Otorchmizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=500)

# Prints out information
print(
    f"Best Fitness: {space.population.best_fitness.item():.6e}"
)
