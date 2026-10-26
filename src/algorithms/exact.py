import time
import numpy as np
from contracts import SolveResult
from algorithms.base import Algorithm
from backends.cupy_backend import require_cupy, to_cupy_csr

class ExactMaxCut(Algorithm):

    name = "exact"
    
    def __init__(self, max_nodes: int = 24, gpu_batch_size: int = 4096):
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes must be an integer >= 1.")
        if isinstance(gpu_batch_size, bool) or not isinstance(gpu_batch_size, int) or gpu_batch_size < 1:
            raise ValueError("gpu_batch_size must be an integer >= 1.")
        self.max_nodes = max_nodes
        self.gpu_batch_size = gpu_batch_size


    def solve(
        self,
        graph,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware == "cpu":
            return self._solve_with_cpu(graph, seed, time_limit_seconds)
        if hardware == "gpu":
            return self._solve_with_gpu(
                graph, seed, time_limit_seconds, device_index=int(params.get("device_index", 0))
            )
        raise ValueError(f"Hardware not supported by {self.name}: {hardware}")


    def _solve_with_cpu(self, graph, seed: int, time_limit_seconds: float | None = None) -> SolveResult:
        n = graph.n_variables
        if n > self.max_nodes:
            raise ValueError(f"Exact supports maximum {self.max_nodes} nodes; received {n}.")
        
        start = time.perf_counter()
        best_solution = np.ones(n, dtype=np.int8)
        best_value, evaluated = float(graph.evaluate(best_solution)), 0

        # La simmetria globale permette di fissare l'ultimo spin a +1
        for mask in range(1 << max(0, n - 1)):
            if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
                break
        
            sol = np.ones(n, dtype=np.int8)
            for bit in range(n - 1):
                if mask & (1 << bit):
                    sol[bit] = -1
        
            value = graph.evaluate(sol)
            evaluated += 1
        
            if value > best_value:
                best_value, best_solution = value, sol.copy()
        
        elapsed = time.perf_counter() - start
        status = "completed" if evaluated == (1 << max(0, n - 1)) else "time_limit"
        
        return SolveResult(self.name, "cpu", best_solution, float(best_value), elapsed,
                            status, seed, {"evaluated_solutions": evaluated, "optimality_certified": status == "completed"})

    def _solve_with_gpu(
        self,
        graph,
        seed: int,
        time_limit_seconds: float | None,
        *,
        device_index: int,
    ) -> SolveResult:
        n = graph.n_variables
        if n > self.max_nodes:
            raise ValueError(f"Exact supports maximum {self.max_nodes} nodes; received {n}.")

        start = time.perf_counter()
        cp, device_name = require_cupy(device_index)
        weights = to_cupy_csr(graph, cp)
        total_candidates = 1 << max(0, n - 1)
        total_weight = float(sum(weight for _, _, weight in graph.iter_edges()))
        best_solution = np.ones(n, dtype=np.int8)
        best_value = float(graph.evaluate(best_solution))
        evaluated = 0
        bit_positions = cp.arange(max(0, n - 1), dtype=cp.uint64)

        for batch_start in range(0, total_candidates, self.gpu_batch_size):
            if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
                break
            batch_end = min(batch_start + self.gpu_batch_size, total_candidates)
            masks = cp.arange(batch_start, batch_end, dtype=cp.uint64)
            candidates = cp.ones((batch_end - batch_start, n), dtype=cp.float64)
            if n > 1:
                bits = (masks[:, None] >> bit_positions[None, :]) & 1
                candidates[:, : n - 1] = 1.0 - 2.0 * bits
            neighbor_sums = weights @ candidates.T
            quadratic = cp.sum(candidates.T * neighbor_sums, axis=0)
            values = 0.5 * total_weight - 0.25 * quadratic
            batch_best_index = int(cp.argmax(values).item())
            batch_best_value = float(values[batch_best_index].item())
            if batch_best_value > best_value:
                best_solution = cp.asnumpy(candidates[batch_best_index]).astype(np.int8)
                best_value = batch_best_value
            evaluated += batch_end - batch_start

        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - start
        status = "completed" if evaluated == total_candidates else "time_limit"
        return SolveResult(
            self.name,
            "gpu",
            best_solution,
            float(graph.evaluate(best_solution)),
            elapsed,
            status,
            seed,
            {
                "evaluated_solutions": evaluated,
                "optimality_certified": status == "completed",
                "execution_strategy": "batched_exhaustive_search",
                "gpu_batch_size": self.gpu_batch_size,
                "device_index": device_index,
                "device": device_name,
            },
        )
