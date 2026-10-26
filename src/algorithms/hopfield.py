"""Asynchronous Hopfield-network dynamics for weighted Max-Cut."""
from __future__ import annotations

import time

import numpy as np

from algorithms.base import Algorithm
from contracts import SolveResult
from problems.maxcut import MaxCutProblem


class HopfieldNetwork(Algorithm):
    """Asynchronous zero-threshold Hopfield dynamics with random neuron order.

    For Max-Cut, the Hopfield coupling matrix is ``-graph.weights``. A neuron
    update therefore flips a spin exactly when that flip strictly increases
    the cut objective. Ties retain the current spin, preventing two-cycles.
    """

    name = "hopfield"

    def __init__(self, max_iterations: int = 100_000) -> None:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations deve essere un intero >= 1.")
        self.max_iterations = max_iterations

    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware != "cpu":
            raise ValueError(f"Hardware not supported by {self.name}: {hardware}")

        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        solution = rng.choice(np.array([-1, 1], dtype=np.int8), size=graph.n_variables)
        neuron_updates = 0
        flips = 0
        sweeps = 0
        converged = False
        timed_out = False

        while neuron_updates < self.max_iterations:
            changed_this_sweep = False
            for node in rng.permutation(graph.n_variables):
                if neuron_updates >= self.max_iterations:
                    break
                if (
                    time_limit_seconds is not None
                    and time.perf_counter() - start >= time_limit_seconds
                ):
                    timed_out = True
                    break

                node_index = int(node)
                delta = graph.flip_delta(solution, node_index)
                neuron_updates += 1
                # With J=-W, asynchronous Hopfield sign updates improve Max-Cut.
                if delta > 0.0:
                    solution[node_index] *= -1
                    flips += 1
                    changed_this_sweep = True

            if timed_out or neuron_updates >= self.max_iterations:
                break

            sweeps += 1
            if not changed_this_sweep:
                converged = True
                break

        elapsed = time.perf_counter() - start
        status = "time_limit" if timed_out else "completed"
        return SolveResult(
            self.name,
            hardware,
            solution,
            float(graph.evaluate(solution)),
            elapsed,
            status,
            seed,
            {
                "dynamics": "asynchronous_random_order",
                "neuron_updates": neuron_updates,
                "flips": flips,
                "sweeps": sweeps,
                "max_iterations": self.max_iterations,
                "converged": converged,
                "coupling": "negative_maxcut_weights",
            },
        )
