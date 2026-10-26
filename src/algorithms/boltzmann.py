"""Fixed-temperature Boltzmann-machine solver using single-spin Gibbs updates."""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.special import expit

from algorithms.base import Algorithm
from contracts import SolveResult
from problems.maxcut import MaxCutProblem


class BoltzmannMachine(Algorithm):
    """Sample Max-Cut states with asynchronous heat-bath/Gibbs dynamics.

    The Ising energy uses couplings ``J=-W``, making lower network energy
    correspond to higher Max-Cut value. Unlike simulated annealing, this
    implementation samples at one fixed temperature and uses conditional
    Gibbs resampling rather than Metropolis proposals.
    """

    name = "boltzmann"

    def __init__(self, steps: int = 50_000, temperature: float = 1.0) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps deve essere un intero >= 1.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature deve essere un numero finito > 0.")
        self.steps = steps
        self.temperature = float(temperature)

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
        if graph.n_variables < 1:
            raise ValueError("Boltzmann Machine requires a graph with at least one node.")

        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        current = rng.choice(np.array([-1, 1], dtype=np.int8), size=graph.n_variables)
        current_value = float(graph.evaluate(current))
        best = current.copy()
        best_value = current_value
        best_step = 0
        updates = 0
        timed_out = False

        for step in range(1, self.steps + 1):
            if (
                time_limit_seconds is not None
                and time.perf_counter() - start >= time_limit_seconds
            ):
                timed_out = True
                break

            node = int(rng.integers(graph.n_variables))
            if sparse.issparse(graph.weights):
                local_field = float(graph.weights.getrow(node).dot(current)[0])
            else:
                local_field = float(np.dot(graph.weights[node], current))

            # P(s_i=+1 | s_-i) = sigmoid(-2 * sum_j w_ij s_j / T)
            probability_plus = float(expit(-2.0 * local_field / self.temperature))
            new_spin = 1 if rng.random() < probability_plus else -1
            updates += 1

            if new_spin != current[node]:
                delta = graph.flip_delta(current, node)
                current[node] = new_spin
                current_value += delta

                if current_value > best_value:
                    best_value = current_value
                    best = current.copy()
                    best_step = step

        elapsed = time.perf_counter() - start
        status = "time_limit" if timed_out else "completed"
        verified_best_value = float(graph.evaluate(best))
        return SolveResult(
            self.name,
            hardware,
            best,
            verified_best_value,
            elapsed,
            status,
            seed,
            {
                "dynamics": "asynchronous_single_spin_gibbs",
                "steps_requested": self.steps,
                "gibbs_updates": updates,
                "temperature": self.temperature,
                "best_step": best_step,
                "best_objective_value": verified_best_value,
                "final_chain_objective_value": float(graph.evaluate(current)),
                "coupling": "negative_maxcut_weights",
            },
        )
