"""Constants used across the Otorchmizer package."""

import torch

# Prevents division by zero, zero logarithms, and numerical errors
EPSILON = 1e-32

# Initial fitness value for all agents (minimization).
# Uses the maximum finite value for float32 to match PyTorch default dtype.
FLOAT_MAX = torch.finfo(torch.float32).max

# Speed of light constant (used by Relativistic PSO)
LIGHT_SPEED = 3e5

# Number of arguments per GP function node
FUNCTION_N_ARGS = {
    "SUM": 2,
    "SUB": 2,
    "MUL": 2,
    "DIV": 2,
    "EXP": 1,
    "SQRT": 1,
    "LOG": 1,
    "ABS": 1,
    "SIN": 1,
    "COS": 1,
}

# Test pass threshold for integration tests
TEST_EPSILON = 100
