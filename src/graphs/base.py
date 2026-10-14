import networkx as nx
import numpy as np

from problems.maxcut import MaxCutProblem


def to_networkx(problem: MaxCutProblem) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(problem.n_variables))
    for i in range(problem.n_variables):
        for j in range(i + 1, problem.n_variables):
            weight = float(problem.weights[i, j])
            if weight != 0:
                graph.add_edge(i, j, weight=weight)
    return graph
