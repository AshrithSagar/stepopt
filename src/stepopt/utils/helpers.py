"""
Helpers
=======
src/stepopt/utils/helpers.py
"""

from typing import Any

from stepopt.constrained.constraint import AbstractConstraint
from stepopt.core.base import IterativeOptimiser
from stepopt.core.info import RunInfo, StepInfo
from stepopt.core.oracle import Oracle
from stepopt.core.stopping import StoppingCriterionType
from stepopt.functions.protocol import ZeroOrderFunctionProto
from stepopt.problems import UnconstrainedProblem
from stepopt.types import Vector


def optimise[F: ZeroOrderFunctionProto](
    objective: F,
    oracle: type[Oracle[F]],
    method: IterativeOptimiser[StepInfo[Oracle[F]]],
    x0: Vector,
    constaint: AbstractConstraint[Any] | None = None,
    criteria: StoppingCriterionType[StepInfo[Oracle[F]]] | None = None,
    show_params: bool = True,
) -> RunInfo[StepInfo[Oracle[F]]] | None:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
