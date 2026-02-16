import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.swarm import PSO

# Random seed for experimental consistency
torch.manual_seed(0)


def sphere(x):
    return (x ** 2).sum(dim=(-1, -2))


# Number of agents and decision variables
n_agents = 20
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound, device="cpu")
space.build()
optimizer = PSO()
function = Function(sphere)

# Bundles every piece into Otorchmizer class
opt = Otorchmizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=1000)

# Prints out information about the best solution found
print(
    f"Best Position: {space.population.best_position.squeeze().tolist()} | "
    f"Fitness: {space.population.best_fitness.item():.6e}"
)
