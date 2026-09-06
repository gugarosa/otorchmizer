# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Constants used across the Otorchmizer package."""

import torch

# Prevents division by zero, zero logarithms, and numerical errors
EPSILON = 1e-32

# Retained as the public finite float32 limit, not a dtype-independent fitness sentinel
FLOAT_MAX = torch.finfo(torch.float32).max

# Relativistic PSO uses the speed of light in kilometers per second
LIGHT_SPEED = 3e5

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

TEST_EPSILON = 100
