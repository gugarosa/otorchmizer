import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.evolutionary import GA

# Random seed for experimental consistency
torch.manual_seed(0)


# Rosenbrock function — a classic non-convex optimization benchmark
def rosenbrock(x):
    x_flat = x.squeeze(-1)
    return ((100 * (x_flat[1:] - x_flat[:-1] ** 2) ** 2) + (1 - x_flat[:-1]) ** 2).sum()


# Number of agents and decision variables
n_agents = 50
n_variables = 10

# Lower and upper bounds
lower_bound = [-5] * n_variables
upper_bound = [10] * n_variables

# Creates the space, optimizer and function
space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound, device="cpu")
space.build()
optimizer = GA(params={"p_selection": 0.75, "p_mutation": 0.25, "p_crossover": 0.5})
function = Function(rosenbrock)

# Bundles every piece into Otorchmizer class
opt = Otorchmizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=500)

# Prints out information about the best solution found
print(
    f"Best Position: {space.population.best_position.squeeze().tolist()[:5]}... | "
    f"Fitness: {space.population.best_fitness.item():.4f}"
)
