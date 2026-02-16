from otorchmizer.optimizers.swarm import ABC, BA, CS, SSA, SCA

# Artificial Bee Colony
abc = ABC(params={"n_trials": 10})
print(f"Algorithm: {abc.algorithm}")

# Bat Algorithm
ba = BA(params={"f_min": 0.0, "f_max": 2.0, "A": 0.5, "r": 0.5})
print(f"Algorithm: {ba.algorithm}")

# Cuckoo Search
cs = CS(params={"alpha": 0.01, "p": 0.25})
print(f"Algorithm: {cs.algorithm}")

# Salp Swarm Algorithm
ssa = SSA()
print(f"Algorithm: {ssa.algorithm}")

# Sine Cosine Algorithm
sca = SCA(params={"r_min": 0.0, "a": 2.0})
print(f"Algorithm: {sca.algorithm}")
