"""Erdos-Renyi graph generators returning the domain MaxCutProblem."""

import networkx as nx
import numpy as np
from scipy import sparse

from problems.maxcut import MaxCutProblem

def erdos_renyi(
    n: int,
    p: float,
    seed: int = 0,
    weighted: bool = False,
    weight_low: float = 1.0,
    weight_high: float = 10.0,
) -> MaxCutProblem:
    if n < 0:
        raise ValueError("n must be non-negative.")

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in range 0-1.")

    if weighted and not (0 < weight_low <= weight_high):
        raise ValueError("Invalid weights range")
    
    graph = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    rng = np.random.default_rng(seed)
    rows, columns, values = [], [], []
    for u, v in graph.edges():
        weight = (float(rng.uniform(weight_low, weight_high)) if weighted else 1.0)
        rows.extend((u, v))
        columns.extend((v, u))
        values.extend((weight, weight))

    weights = sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()

    return MaxCutProblem(weights=weights)


def erdos_renyi_raw(
    n: int,
    p: float,
    seed: int = 0,
    weighted: bool = False,
    weight_low: float = 1.0,
    weight_high: float = 10.0,
) -> MaxCutProblem:
    if n < 0:
        raise ValueError("n must be non-negative.")

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in range 0-1.")

    if weighted and not (0 < weight_low <= weight_high):
        raise ValueError("Invalid weights range")

    rng = np.random.default_rng(seed)
    weights = np.zeros((n, n), dtype=np.float64)

    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                weight = (float(rng.uniform(weight_low, weight_high)) if weighted else 1.0)
                weights[u, v] = weight
                weights[v, u] = weight

    return MaxCutProblem(weights=weights)
