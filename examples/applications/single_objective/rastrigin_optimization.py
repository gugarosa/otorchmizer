import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.swarm import FA

# Random seed for experimental consistency
torch.manual_seed(0)


def rastrigin(x):
    n = x.shape[0]
    return 10 * n + (x ** 2 - 10 * torch.cos(2 * torch.pi * x)).sum(dim=(-1, -2))


# Number of agents and decision variables
n_agents = 30
n_variables = 5

# Lower and upper bounds
lower_bound = [-5.12] * n_variables
upper_bound = [5.12] * n_variables

# Creates the space, optimizer and function
space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound, device="cpu")
space.build()
optimizer = FA(params={"alpha": 0.5, "beta": 0.2, "gamma": 1.0})
function = Function(rastrigin)

# Bundles every piece into Otorchmizer class
opt = Otorchmizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=200)

# Prints out information about the best solution found
print(
    f"Best Position: {space.population.best_position.squeeze().tolist()[:3]}... | "
    f"Fitness: {space.population.best_fitness.item():.6e}"
)
