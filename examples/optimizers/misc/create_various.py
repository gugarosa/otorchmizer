from otorchmizer.optimizers.misc import HC, GS, CEM

# Hill Climbing
hc = HC(params={"r_mean": 0.0, "r_var": 0.1})
print(f"Algorithm: {hc.algorithm}")

# Grid Search
gs = GS(params={"step": 0.1})
print(f"Algorithm: {gs.algorithm}")

# Cross-Entropy Method
cem = CEM()
print(f"Algorithm: {cem.algorithm}")
