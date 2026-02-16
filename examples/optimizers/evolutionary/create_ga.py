from otorchmizer.optimizers.evolutionary import GA

# Creates a Genetic Algorithm optimizer
params = {"p_selection": 0.75, "p_mutation": 0.25, "p_crossover": 0.5}
o = GA(params=params)

# Prints out some properties
print(f"Algorithm: {o.algorithm}")
print(f"Parameters: {o.params}")
