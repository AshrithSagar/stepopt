"""
Type aliases and utility types
=======
src/stepopt/types.py
"""

from typing import TypeAlias

import numpy as np
from typingkit.numpy._typed.helpers import Array1D, Array2D

## Helpers

dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""

Scalar: TypeAlias = float
"""A type alias for a scalar real number."""

Vector: TypeAlias = Array1D[int]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""

Matrix: TypeAlias = Array2D[int, int]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""
