"""
Stopping criteria
=======
src/stepopt/core/stopping.py
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import numpy as np

from stepopt.core.info import FirstOrderStepInfo, StepInfo, ZeroOrderStepInfo
from stepopt.core.oracle import FirstOrderOracle, Oracle, ZeroOrderOracle
from stepopt.functions.protocol import (
    FirstOrderFunctionProto,
    FunctionProto,
    ZeroOrderFunctionProto,
)
from stepopt.types import Scalar
from stepopt.utils.logging import logger

type StoppingCriterionType[S: StepInfo[Oracle[FunctionProto]]] = (
    StoppingCriterion[S] | CompositeCriterion[S] | Iterable[StoppingCriterion[S]]
)
"""Generic type alias for stopping criteria."""


class StoppingCriterion[S: StepInfo[Oracle[FunctionProto]]](ABC):
    """An abstract base class to encapsulate various stopping criteria for iterative algorithms."""

    def reset(self) -> None:
        """Reset internal state, if any. Called at the beginning of each run."""
        name = self.__class__.__name__
        logger.debug(f"Stopping criterion [yellow]{name}[/] has been reset.")

    @abstractmethod
    def check(self, info: S) -> bool:
        """
        Return True if the stopping criterion is met.
        [Required]: This method should be implemented by subclasses to define the specific stopping condition.
        Parameters:
            x: Current value of `x`, i.e., `x_k`.
            k: Current iteration number.
            oracle_fn: The oracle function to query for `f(x)`.
        """
        raise NotImplementedError

    @override
    def __str__(self) -> str:
        name = self.__class__.__name__
        params = self.__dict__ if self.__dict__ else ""
        return f"{name}({params})"


@dataclass
class CompositeCriterion[S: StepInfo[Oracle[FunctionProto]]](StoppingCriterion[S]):
    """
    Combines multiple stopping criteria. Stops when any one of the criteria is met.
    """

    criteria: Iterable[StoppingCriterion[S]]
    """Iterable of stopping criteria."""

    @override
    def reset(self) -> None:
        for criterion in self.criteria:
            criterion.reset()

    @override
    def check(self, info: S) -> bool:
        logger.debug(
            f"Checking stopping criteria for {info.__class__.__name__}(k={info.k})"
        )
        return any(criterion.check(info) for criterion in self.criteria)


@dataclass
class MaxIterationsCriterion[S: StepInfo[Oracle[FunctionProto]]](StoppingCriterion[S]):
    """
    Stops when the maximum number of iterations is reached.

    `k >= maxiter`
    """

    maxiter: int = 1000
    """Maximum number of iterations."""

    @override
    def check(self, info: S) -> bool:
        return bool(info.k >= self.maxiter)


@dataclass
class GradientNormCriterion[
    S: FirstOrderStepInfo[FirstOrderOracle[FirstOrderFunctionProto]]
](StoppingCriterion[S]):
    """
    Stops when the gradient norm is below a specified tolerance.

    `||f'(x_k)|| < tol`
    """

    tol: Scalar = 1e-6
    """Tolerance for the gradient norm."""

    @override
    def check(self, info: S) -> bool:
        return bool(np.linalg.norm(info.dfx) < self.tol)


@dataclass
class FunctionValueCriterion[
    S: ZeroOrderStepInfo[ZeroOrderOracle[ZeroOrderFunctionProto]]
](StoppingCriterion[S]):
    """
    Stops when the function value is below a specified tolerance.

    `f(x_k) < tol`
    """

    tol: Scalar = 1e-6
    """Tolerance for the function value."""

    @override
    def check(self, info: S) -> bool:
        return bool(info.fx < self.tol)
