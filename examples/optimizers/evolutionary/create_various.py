from otorchmizer.optimizers.evolutionary import DE, HS, IHS

# Differential Evolution
de = DE(params={"CR": 0.9, "F": 0.7})
print(f"Algorithm: {de.algorithm}")

# Harmony Search
hs = HS(params={"HMCR": 0.7, "PAR": 0.7, "bw": 1.0})
print(f"Algorithm: {hs.algorithm}")

# Improved Harmony Search
ihs = IHS(params={"HMCR": 0.7, "PAR_min": 0.0, "PAR_max": 1.0, "bw_min": 1.0, "bw_max": 10.0})
print(f"Algorithm: {ihs.algorithm}")
