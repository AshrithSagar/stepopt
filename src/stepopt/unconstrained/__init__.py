"""
Unconstrained optimisers implementations
=======
src/stepopt/unconstrained/__init__.py

References
-------
- Nocedal, J., & Wright, S. J. (2006). Numerical optimization. Springer.
"""

from stepopt.unconstrained.conjugate_gradient import (
    ConjugateDirectionMethod,
    ConjugateGradientMethod,
)
from stepopt.unconstrained.line_search import (
    GradientDescent,
    GradientDescentArmijo,
    GradientDescentArmijoGoldstein,
    GradientDescentBacktracking,
    GradientDescentExactLineSearch,
    GradientDescentWolfe,
)
from stepopt.unconstrained.quasi_newton import (
    BFGSUpdate,
    DFPUpdate,
    NewtonMethod,
    SR1Update,
)

__all__ = [
    "ConjugateDirectionMethod",
    "ConjugateGradientMethod",
    "GradientDescent",
    "GradientDescentArmijo",
    "GradientDescentArmijoGoldstein",
    "GradientDescentBacktracking",
    "GradientDescentExactLineSearch",
    "GradientDescentWolfe",
    "BFGSUpdate",
    "DFPUpdate",
    "NewtonMethod",
    "SR1Update",
]
