import time
import numpy as np
from contracts import SolveResult
from algorithms.base import Algorithm

class ExactMaxCut(Algorithm):

    name = "exact"
    
    def __init__(self, max_nodes: int = 24):
        self.max_nodes = max_nodes


    def solve(self, graph, hardware: str, seed: int, time_limit_seconds: float | None = None) -> SolveResult:
        if hardware == "cpu":
            return self._solve_with_cpu(graph, seed, time_limit_seconds)
        
        if hardware == "gpu":
            return self._solve_with_gpu()

        return SolveResult()


    def _solve_with_cpu(self, graph, seed: int, time_limit_seconds: float | None = None) -> SolveResult:
        n = graph.n_variables
        if n > self.max_nodes:
            raise ValueError(f"Exact supports maximum {self.max_nodes} nodes; received {n}.")
        
        start = time.perf_counter()
        best_solution, best_value, evaluated = None, float("-inf"), 0

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


    def _solve_with_gpu(self, graph, seed: int, time_limit_seconds: float | None = None) -> SolveResult:
        pass
    