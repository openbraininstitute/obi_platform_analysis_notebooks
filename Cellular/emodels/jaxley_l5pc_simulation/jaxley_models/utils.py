# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from jaxley.solver_gate import save_exp


def sigmoid(v, num, den):
    """Basic sigmoid function used in channel kinetics."""
    return 1.0 / (1.0 + save_exp((v + num) / den))


def double_exp(v, num1, den1, num2, den2):
    """Sum of two exponential terms used in some time constants."""
    return save_exp((v + num1) / den1) + save_exp((v + num2) / den2)


def temp_factor(q10, temp):
    return q10 ** ((temp - 283.0) / 10.0)
