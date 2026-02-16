from otorchmizer.optimizers.swarm import FA

# Creates a Firefly Algorithm optimizer
params = {"alpha": 0.5, "beta": 0.2, "gamma": 1.0}
o = FA(params=params)

# Prints out some properties
print(f"Algorithm: {o.algorithm}")
print(f"Parameters: {o.params}")
