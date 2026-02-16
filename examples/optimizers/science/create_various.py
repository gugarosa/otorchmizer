from otorchmizer.optimizers.science import SA, GSA, EO

# Simulated Annealing
sa = SA(params={"T": 100.0, "beta": 0.999})
print(f"Algorithm: {sa.algorithm}")

# Gravitational Search Algorithm
gsa = GSA(params={"G": 100.0})
print(f"Algorithm: {gsa.algorithm}")

# Equilibrium Optimizer
eo = EO(params={"a1": 2.0, "a2": 1.0, "GP": 0.5, "V": 1.0})
print(f"Algorithm: {eo.algorithm}")
