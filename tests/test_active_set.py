"""
tests/test_active_set.py
"""

import numpy as np
from stepopt.constrained.constraint import LinearInequalityConstraintSet
from stepopt.constrained.optimisers import ActiveSetMethod
from stepopt.core.oracle import FirstOrderOracle
from stepopt.core.stopping import MaxIterationsCriterion
from stepopt.functions import ConvexQuadratic
from stepopt.problems import InequalityConstrainedQuadraticProgram
from stepopt.types import Matrix, Vector
from stepopt.utils.logging import Logger, console

Logger.configure(level="DEBUG")


def test_active_set() -> None:
    dim = int(2)
    Q = Matrix([[3, 0], [0, 2]])
    h = Vector([-2, -5])
    func = ConvexQuadratic(dim=dim, Q=Q, h=h)

    A_ineq = Matrix([[-1, 2], [1, 2], [2, 1], [1, -1]])
    b_ineq = Vector([2, 6, 6, 2])
    constraint = LinearInequalityConstraintSet(A=A_ineq, b=b_ineq)
    problem = InequalityConstrainedQuadraticProgram(
        objective=func, oracle=FirstOrderOracle, constraint=constraint
    )

    optimiser = ActiveSetMethod(problem)
    oracle = FirstOrderOracle(func)
    x0 = Vector(np.zeros(dim))
    criteria = MaxIterationsCriterion(maxiter=int(1e2))

    assert constraint.is_satisfied(x0), "Initial point is not feasible."
    info = optimiser.run(oracle_fn=oracle, x0=x0, criteria=criteria)
    console.print(str(info))


if __name__ == "__main__":
    test_active_set()
