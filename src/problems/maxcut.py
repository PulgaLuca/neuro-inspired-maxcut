from dataclasses import dataclass
import numpy as np

@dataclass
class MaxCutProblem:
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)

        if self.weights.ndim != 2:
            raise ValueError("Matrix must be bi-dimensional.")

        n_rows, n_colums = self.weights.shape
        if n_rows != n_colums:
            raise ValueError("Matrix must be quadratic.")

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Matrix weights must be finite.")

        if np.any(self.weights < 0):
            raise ValueError("This version could not handle non-positive weights.")

        if not np.allclose(self.weights, self.weights.T):
            raise ValueError("Matrix must be simmetric.")

        if not np.allclose(np.diag(self.weights), 0):
            raise ValueError("Graph must not contain self-loops.")

    @property
    def n_variables(self) -> int:
        return self.weights.shape[0] # -> n_rows

    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate the total weight given by summing all the cut-weights."""
        s = np.asarray(solution)

        if s.shape != (self.n_variables,):
            raise ValueError("Invalid solution dimension.")

        if not np.all(np.isin(s, [-1, 1])):
            raise ValueError("Each variable must have spin equal to -1 or +1.")

        i, j = np.triu_indices(self.n_variables, k=1)
        return float(np.sum(self.weights[i, j] * (1-s[i]*s[j]) / 2))
    