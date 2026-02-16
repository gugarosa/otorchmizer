import torch

from otorchmizer import Otorchmizer
from otorchmizer.core import Function, Space
from otorchmizer.optimizers.boolean import BPSO

# Random seed for experimental consistency
torch.manual_seed(0)


# Simple binary fitness function — count number of 1s
def count_ones(x):
    return -x.sum()  # Minimize negative count = maximize count of 1s


# Number of agents and decision variables
n_agents = 20
n_variables = 10

# Binary bounds
lower_bound = [0] * n_variables
upper_bound = [1] * n_variables

# Creates the space, optimizer and function
space = Space(n_agents=n_agents, n_variables=n_variables,
              lower_bound=lower_bound, upper_bound=upper_bound, device="cpu")

# Initialize with binary positions
space.population.initialize_binary()
space._built = True  # Mark as built after manual init

optimizer = BPSO(params={"c1": 1.7, "c2": 1.7})
function = Function(count_ones)

# Bundles every piece into Otorchmizer class
opt = Otorchmizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=100)

# Prints out information about the best solution found
print(
    f"Best Position: {space.population.best_position.squeeze().tolist()} | "
    f"Fitness: {space.population.best_fitness.item():.4f}"
)
