"""Small state-vector QAOA simulator for Max-Cut (classical CPU execution)."""
from __future__ import annotations

import time

import numpy as np

from algorithms.base import Algorithm
from contracts import SolveResult
from problems.maxcut import MaxCutProblem


class QAOAMaxCut(Algorithm):
    """Optimize a depth-p QAOA circuit using exact CPU state-vector simulation.

    This is a classical simulator, not a quantum processor or a claim of
    quantum speedup. The exponential state vector intentionally limits this
    reference implementation to small graphs.
    """

    name = "qaoa"

    def __init__(
        self,
        depth: int = 1,
        optimizer_iterations: int = 80,
        max_nodes: int = 20,
        optimizer_restarts: int = 1,
        shots: int = 1024,
    ) -> None:
        for name, value in (
            ("depth", depth),
            ("optimizer_iterations", optimizer_iterations),
            ("max_nodes", max_nodes),
            ("optimizer_restarts", optimizer_restarts),
            ("shots", shots),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        self.depth = depth
        self.optimizer_iterations = optimizer_iterations
        self.max_nodes = max_nodes
        self.optimizer_restarts = optimizer_restarts
        self.shots = shots

    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware != "quantum_simulator":
            raise ValueError(
                "QAOA reference implementation supports only 'quantum_simulator'; "
                "a physical quantum provider is not configured."
            )
        n = graph.n_variables
        if n > self.max_nodes:
            raise ValueError(
                f"QAOA state-vector simulation supports at most {self.max_nodes} "
                f"nodes; received {n}."
            )

        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        dimension = 1 << n
        states = np.arange(dimension, dtype=np.uint64)
        bit_indices = np.arange(n, dtype=np.uint64)
        # Bit 0 maps to spin +1, bit 1 to spin -1.
        spins = np.where(((states[:, None] >> bit_indices) & 1) == 0, 1, -1)
        edge_u, edge_v, edge_w = _edge_arrays(graph)
        cut_values = np.zeros(dimension, dtype=np.float64)
        if len(edge_w):
            cut_values = np.sum(
                edge_w[None, :] * (1 - spins[:, edge_u] * spins[:, edge_v]) / 2,
                axis=1,
            )

        best_index = int(rng.integers(dimension))
        best_value = float(cut_values[best_index])
        best_angles: list[float] = []
        evaluations = 0
        timed_out = False

        for _restart in range(self.optimizer_restarts):
            angles = rng.uniform(0.0, 2 * np.pi, size=2 * self.depth)
            current_energy, probabilities = _qaoa_probabilities(
                n, edge_u, edge_v, edge_w, angles, self.depth
            )
            evaluations += 1
            # Seeded, derivative-free coordinate search keeps the implementation
            # dependency-free while providing a usable small-instance baseline.
            step = np.pi / 2
            for _iteration in range(self.optimizer_iterations):
                if (
                    time_limit_seconds is not None
                    and time.perf_counter() - start >= time_limit_seconds
                ):
                    timed_out = True
                    break
                improved = False
                for coordinate in range(2 * self.depth):
                    for direction in (-1.0, 1.0):
                        candidate = angles.copy()
                        candidate[coordinate] = (
                            candidate[coordinate] + direction * step
                        ) % (2 * np.pi)
                        energy, candidate_probabilities = _qaoa_probabilities(
                            n, edge_u, edge_v, edge_w, candidate, self.depth
                        )
                        evaluations += 1
                        if energy < current_energy - 1e-12:
                            angles = candidate
                            current_energy = energy
                            probabilities = candidate_probabilities
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-3:
                        break
            sampled_indices = rng.choice(dimension, size=self.shots, p=probabilities)
            sampled_index = int(sampled_indices[np.argmax(cut_values[sampled_indices])])
            sampled_value = float(cut_values[sampled_index])
            if sampled_value > best_value:
                best_value = sampled_value
                best_index = sampled_index
                best_angles = angles.tolist()
            if timed_out:
                break

        solution = spins[best_index].astype(np.int8)
        objective = float(graph.evaluate(solution))
        elapsed = time.perf_counter() - start
        return SolveResult(
            self.name,
            hardware,
            solution,
            objective,
            elapsed,
            "time_limit" if timed_out else "completed",
            seed,
            {
                "ansatz": "qaoa",
                "depth": self.depth,
                "optimizer": "seeded_coordinate_search",
                "optimizer_iterations": self.optimizer_iterations,
                "optimizer_restarts": self.optimizer_restarts,
                "shots": self.shots,
                "objective_evaluations": evaluations,
                "optimized_angles": best_angles,
                "statevector_dimension": dimension,
                "provider": "numpy_statevector_simulator",
                "device": "CPU state-vector simulation",
                "quantum_hardware_used": False,
                "best_objective_value": objective,
            },
        )


def _edge_arrays(graph: MaxCutProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = list(graph.iter_edges())
    if not edges:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    u, v, weights = zip(*edges, strict=True)
    return (
        np.asarray(u, dtype=np.int64),
        np.asarray(v, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def _qaoa_probabilities(
    n: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    angles: np.ndarray,
    depth: int,
) -> tuple[float, np.ndarray]:
    state = np.full(1 << n, (1 / np.sqrt(1 << n)), dtype=np.complex128)
    states = np.arange(1 << n, dtype=np.uint64)
    bit_indices = np.arange(n, dtype=np.uint64)
    z = np.where(((states[:, None] >> bit_indices) & 1) == 0, 1.0, -1.0)
    cut = np.zeros(1 << n, dtype=np.float64)
    if len(edge_w):
        cut = np.sum(edge_w[None, :] * (1 - z[:, edge_u] * z[:, edge_v]) / 2, axis=1)

    for layer in range(depth):
        gamma, beta = angles[2 * layer], angles[2 * layer + 1]
        state *= np.exp(1j * gamma * cut)
        for qubit in range(n):
            stride = 1 << qubit
            blocks = state.reshape(-1, 2, stride)
            zero = blocks[:, 0, :].copy()
            one = blocks[:, 1, :].copy()
            cosine, sine = np.cos(beta), np.sin(beta)
            blocks[:, 0, :] = cosine * zero - 1j * sine * one
            blocks[:, 1, :] = -1j * sine * zero + cosine * one

    probabilities = np.abs(state) ** 2
    probabilities /= probabilities.sum()
    expected_cut = float(np.dot(probabilities, cut))
    return -expected_cut, probabilities
