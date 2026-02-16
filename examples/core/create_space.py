from otorchmizer.core import Space

# Creates a search space
# device="auto" picks GPU if available, otherwise CPU
space = Space(n_agents=20, n_variables=5,
              lower_bound=[-10, -10, -10, -10, -10],
              upper_bound=[10, 10, 10, 10, 10],
              device="cpu")

# Builds the space (initializes population)
space.build()

# Prints out some properties
print(f"Population shape: {space.population.positions.shape}")
print(f"Device: {space.population.device}")
print(f"Bounds: [{space.population.lb.squeeze().tolist()}, {space.population.ub.squeeze().tolist()}]")
