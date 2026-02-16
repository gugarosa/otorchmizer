from otorchmizer.optimizers.social import BSO, QSA

# Brain Storm Optimization
bso = BSO(params={"k": 5})
print(f"Algorithm: {bso.algorithm}")

# Queuing Search Algorithm
qsa = QSA()
print(f"Algorithm: {qsa.algorithm}")
