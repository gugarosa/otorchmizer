import torch

from otorchmizer.core import Population

# We need to define the amount of decision variables
# and its dimension (single, complex, quaternion, octonion, sedenion)
n_variables = 2
n_dimensions = 1

# We also need to define its bounds
lower_bound = torch.tensor([0.0, 0.0])
upper_bound = torch.tensor([1.0, 1.0])

# Creates a new Population (replaces Agent from Opytimizer)
pop = Population(n_agents=10, n_variables=n_variables,
                 n_dimensions=n_dimensions,
                 lower_bound=lower_bound, upper_bound=upper_bound)

# Initializes positions uniformly within bounds
pop.initialize_uniform()

# Prints out some properties
print(f"Population: {pop}")
print(f"Positions shape: {pop.positions.shape}")
print(f"First agent position: {pop.positions[0].squeeze()}")
print(f"Best fitness: {pop.best_fitness.item()}")
