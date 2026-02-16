from otorchmizer.optimizers.boolean import BPSO, UMDA

# Binary Particle Swarm Optimization
bpso = BPSO(params={"c1": 1.7, "c2": 1.7})
print(f"Algorithm: {bpso.algorithm}")

# Univariate Marginal Distribution Algorithm
umda = UMDA()
print(f"Algorithm: {umda.algorithm}")
