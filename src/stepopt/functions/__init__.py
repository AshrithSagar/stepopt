"""
Optimisation functions
=======
src/stepopt/functions/__init__.py
"""

from stepopt.functions.base import ConvexQuadratic, Function, LinearFunction
from stepopt.functions.test import (
    Ackley,
    Beale,
    Booth,
    DropWave,
    Eggholder,
    Griewank,
    Levy,
    Levy13,
    Matyas,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    SumSquares,
    ThreeHumpCamel,
    Zakharov,
)

__all__ = [
    "Function",
    "LinearFunction",
    "ConvexQuadratic",
    "Ackley",
    "Beale",
    "Booth",
    "DropWave",
    "Eggholder",
    "Griewank",
    "Levy",
    "Levy13",
    "Matyas",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
    "SumSquares",
    "ThreeHumpCamel",
    "Zakharov",
]
