from dataclasses import dataclass, asdict, field
from typing import Any
import numpy as np


@dataclass
class SolveResult:
    """
    Standardized contract (object container) for algorithms results.
    """
    algorithm: str
    hardware: str
    solution: np.ndarray    # Found solution. In Max-Cut problem, it represents the group assigned to each node.
    objective_value: float  # Objective function. In Max-Cut, it represents the cut-edges' weights.
    runtime_seconds: float  # Runtime execution durantion in seconds, useful to comprare algorithms efficiency.
    status: str             # Final execution status, "success" or "timeout".
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)     # Additional/Optional infos about/used by the algorithm.

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["solution"] = self.solution.astype(int).tolist()
        return result
