"""
Helpers
=======
src/stepopt/utils/helpers.py
"""

from typing import Any, Optional, Type

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
    oracle: Type[Oracle[F]],
    method: IterativeOptimiser[StepInfo[Oracle[F]]],
    x0: Vector,
    constaint: Optional[AbstractConstraint[Any]] = None,
    criteria: Optional[StoppingCriterionType[StepInfo[Oracle[F]]]] = None,
    show_params: bool = True,
) -> Optional[RunInfo[StepInfo[Oracle[F]]]]:
    if constaint is None:
        problem = UnconstrainedProblem(objective, oracle)
        info = problem.solve(method, x0, criteria, show_params)
        return info
    return None
