from dataclasses import dataclass
from typing import Iterator
import numpy as np
from scipy import sparse

@dataclass
class MaxCutProblem:
    weights: np.ndarray | sparse.spmatrix

    def __post_init__(self) -> None:
        if sparse.issparse(self.weights):
            self.weights = self.weights.astype(np.float64).tocsr()
            self.weights.sum_duplicates()
            self.weights.sort_indices()
        else:
            self.weights = np.asarray(self.weights, dtype=np.float64)

        if len(self.weights.shape) != 2:
            raise ValueError("Matrix must be bi-dimensional.")

        n_rows, n_colums = self.weights.shape
        if n_rows != n_colums:
            raise ValueError("Matrix must be quadratic.")

        values = self.weights.data if sparse.issparse(self.weights) else self.weights
        if not np.all(np.isfinite(values)):
            raise ValueError("Matrix weights must be finite.")

        symmetric_difference = self.weights - self.weights.T
        if sparse.issparse(symmetric_difference):
            symmetric = np.allclose(symmetric_difference.data, 0)
        else:
            symmetric = np.allclose(symmetric_difference, 0)
        if not symmetric:
            raise ValueError("Matrix must be simmetric.")

        if not np.allclose(self.weights.diagonal(), 0):
            raise ValueError("Graph must not contain self-loops.")

    @property
    def n_variables(self) -> int:
        return self.weights.shape[0] # -> n_rows

    @property
    def n_edges(self) -> int:
        if sparse.issparse(self.weights):
            return int(sparse.triu(self.weights, k=1).nnz)
        return int(np.count_nonzero(np.triu(self.weights, k=1)))

    def iter_edges(self) -> Iterator[tuple[int, int, float]]:
        """Yield each undirected edge once, preserving its signed weight."""
        if sparse.issparse(self.weights):
            upper = sparse.triu(self.weights, k=1).tocoo()
            yield from (
                (int(u), int(v), float(weight))
                for u, v, weight in zip(upper.row, upper.col, upper.data)
            )
            return

        rows, columns = np.nonzero(np.triu(self.weights, k=1))
        yield from (
            (int(u), int(v), float(self.weights[u, v]))
            for u, v in zip(rows, columns)
        )

    def flip_delta(self, solution: np.ndarray, node: int) -> float:
        """Return the cut-value change from flipping one spin."""
        if sparse.issparse(self.weights):
            neighbor_sum = self.weights.getrow(node).dot(solution)[0]
        else:
            neighbor_sum = np.dot(self.weights[node], solution)
        return float(solution[node] * neighbor_sum)

    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate the total weight given by summing all the cut-weights."""
        s = np.asarray(solution)

        if s.shape != (self.n_variables,):
            raise ValueError("Invalid solution dimension.")

        if not np.all(np.isin(s, [-1, 1])):
            raise ValueError("Each variable must have spin equal to -1 or +1.")

        if sparse.issparse(self.weights):
            upper = sparse.triu(self.weights, k=1).tocoo()
            return float(np.sum(
                upper.data * (1 - s[upper.row] * s[upper.col]) / 2
            ))

        i, j = np.triu_indices(self.n_variables, k=1)
        return float(np.sum(self.weights[i, j] * (1-s[i]*s[j]) / 2))
    