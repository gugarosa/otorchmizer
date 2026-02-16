import torch

from otorchmizer.math import random, distribution

# Generate uniform random numbers
u = random.generate_uniform_random_number(low=0, high=1, size=(5,))
print(f"Uniform: {u}")

# Generate Gaussian random numbers
g = random.generate_gaussian_random_number(mean=0.0, variance=1.0, size=(5,))
print(f"Gaussian: {g}")

# Generate Lévy distribution
levy = distribution.generate_levy_distribution(beta=1.5, size=(5,))
print(f"Lévy: {levy}")

# Generate Bernoulli distribution
bern = distribution.generate_bernoulli_distribution(prob=0.5, size=(5,))
print(f"Bernoulli: {bern}")
