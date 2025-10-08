"""
minigrad: A minimal neural network library with automatic differentiation.

Core modules:
- tensor: Tensor class with autograd support
- nn: Neural network layers (Layer, MLP)
- datasets: Synthetic dataset generators
"""

from .tensor import Tensor
from .nn import Layer, MLP
from .datasets import make_moons, make_circles, make_linear_regression, make_spirals

__version__ = "0.2.0"
__all__ = [
    "Tensor",
    "Layer",
    "MLP",
    "make_moons",
    "make_circles",
    "make_linear_regression",
    "make_spirals",
]
