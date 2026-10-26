import time

import numpy as np

from algorithms.base import Algorithm
from contracts import SolveResult
from backends.cupy_backend import require_cupy, to_cupy_csr


class LocalSearch(Algorithm):
    """First-improvement one-flip local search for Max-Cut."""

    name = "local_search"

    def __init__(self, max_iterations: int = 10000) -> None:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations deve essere un intero >= 1.")
        self.max_iterations = max_iterations

    def solve(
        self,
        graph,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware == "gpu":
            return self._solve_gpu(
                graph, seed, time_limit_seconds,
                device_index=int(params.get("device_index", 0)),
            )
        if hardware != "cpu":
            raise ValueError(f"Hardware not supported by {self.name}: {hardware}")

        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        solution = rng.choice(np.array([-1, 1], dtype=np.int8), size=graph.n_variables)
        iterations = 0
        flip_evaluations = 0
        local_optimum = False
        timed_out = False

        while iterations < self.max_iterations:
            if (
                time_limit_seconds is not None
                and time.perf_counter() - start >= time_limit_seconds
            ):
                timed_out = True
                break

            improved = False
            for node in rng.permutation(graph.n_variables):
                if (
                    time_limit_seconds is not None
                    and time.perf_counter() - start >= time_limit_seconds
                ):
                    timed_out = True
                    break

                node_index = int(node)
                flip_evaluations += 1
                delta = graph.flip_delta(solution, node_index)
                if delta > 0:
                    solution[node_index] *= -1
                    iterations += 1
                    improved = True
                    break

            if timed_out:
                break
            if not improved:
                local_optimum = True
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
                "iterations": iterations,
                "max_iterations": self.max_iterations,
                "flip_evaluations": flip_evaluations,
                "local_optimum": local_optimum,
            },
        )

    def _solve_gpu(
        self,
        graph,
        seed: int,
        time_limit_seconds: float | None,
        *,
        device_index: int,
    ) -> SolveResult:
        """GPU best-improvement one-flip search over all nodes in parallel."""
        start = time.perf_counter()
        cp, device_name = require_cupy(device_index)
        weights = to_cupy_csr(graph, cp)
        rng = cp.random.RandomState(seed)
        solution = rng.choice(cp.asarray([-1, 1], dtype=cp.int8), size=graph.n_variables)
        iterations = 0
        flip_evaluations = 0
        local_optimum = False
        timed_out = False
        while iterations < self.max_iterations:
            if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
                timed_out = True
                break
            deltas = solution * (weights @ solution)
            flip_evaluations += graph.n_variables
            node = int(cp.argmax(deltas).item())
            best_delta = float(deltas[node].item())
            if best_delta <= 0.0:
                local_optimum = True
                break
            solution[node] *= -1
            iterations += 1

        cp.cuda.get_current_stream().synchronize()
        solution_cpu = cp.asnumpy(solution).astype(np.int8)
        elapsed = time.perf_counter() - start
        status = "time_limit" if timed_out else "completed"
        return SolveResult(
            self.name,
            "gpu",
            solution_cpu,
            float(graph.evaluate(solution_cpu)),
            elapsed,
            status,
            seed,
            {
                "iterations": iterations,
                "max_iterations": self.max_iterations,
                "flip_evaluations": flip_evaluations,
                "local_optimum": local_optimum,
                "execution_strategy": "parallel_best_improvement_one_flip",
                "device_index": device_index,
                "device": device_name,
            },
        )
