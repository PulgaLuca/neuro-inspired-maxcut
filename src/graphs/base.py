import networkx as nx
import numpy as np

from problems.maxcut import MaxCutProblem


def to_networkx(problem: MaxCutProblem) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(problem.n_variables))
    graph.add_weighted_edges_from(problem.iter_edges())
    return graph
