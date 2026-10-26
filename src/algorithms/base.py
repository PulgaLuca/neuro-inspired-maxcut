from abc import ABC, abstractmethod
from contracts import SolveResult
from problems.maxcut import MaxCutProblem

class Algorithm(ABC):
    """
    Base abstract class for all the resolution algorithms.

    It defines a unique and simple interface to use different algorithms
    to solve a given problem (in our case MAX-CUT).
    """

    name: str

    @abstractmethod
    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        """
        Solve a given problem and it returns a SolveResult object.

        problem: problem's istance.
        hardware: platform used to run the algorithms (CPU, GPU, etc..).
        seed: 2casuality controller".
        **params: optional parameters for specific algorithms.
        """
        pass
