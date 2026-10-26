import numpy as np

from contracts import SolveResult
from algorithms.base import Algorithm
from problems.maxcut import MaxCutProblem
from backends.cupy_backend import require_cupy, to_cupy_csr
import time

class SimulatedAnnealing(Algorithm):
    name = "simulated_annealing"

    def __init__(
        self,
        steps: int = 10000,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        gpu_replicas: int = 128,
    ):
        if steps < 1: 
            raise ValueError("steps deve essere >= 1.")
        if initial_temperature <= 0 or final_temperature <= 0:
            raise ValueError("Le temperature devono essere positive.")
        if final_temperature > initial_temperature:
            raise ValueError("final_temperature deve essere <= initial_temperature.")
        if isinstance(gpu_replicas, bool) or not isinstance(gpu_replicas, int) or gpu_replicas < 1:
            raise ValueError("gpu_replicas must be an integer >= 1.")
        
        self.steps = steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.gpu_replicas = gpu_replicas

    
    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware == "cpu":
            return self._solve_cpu(graph, seed, time_limit_seconds)
        if hardware == "gpu":
            return self._solve_gpu(
                graph,
                seed,
                time_limit_seconds,
                device_index=int(params.get("device_index", 0)),
            )
        raise ValueError(f"Hardware not supported by {self.name}: {hardware}")
    
    
    def _solve_cpu(self, graph, seed: int, time_limit_seconds: float | None = None) -> SolveResult:
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = graph.n_variables
        current = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        
        current_value = graph.evaluate(current)
        
        best, best_value = current.copy(), current_value
        completed_steps = 0
        
        for step in range(self.steps):
            if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
                break

            fraction = step/max(1, self.steps - 1)
            temperature = self.initial_temperature * (self.final_temperature/self.initial_temperature) ** fraction
            i = int(rng.integers(n))
            # Delta del taglio quando si inverte s_i.
            delta = graph.flip_delta(current, i)
            
            if delta >= 0 or rng.random() < np.exp(delta / max(temperature, 1e-12)):
                current[i] *= -1
                current_value += delta
                if current_value > best_value:
                    best, best_value = current.copy(), current_value
            completed_steps += 1
        
        elapsed = time.perf_counter() - start
        status = "completed" if completed_steps == self.steps else "time_limit"
        
        return SolveResult(self.name, "cpu", best, float(graph.evaluate(best)), elapsed,
                           status, seed, {"steps_requested": self.steps, "steps_completed": completed_steps})

    def _solve_gpu(
        self,
        graph: MaxCutProblem,
        seed: int,
        time_limit_seconds: float | None,
        *,
        device_index: int,
    ) -> SolveResult:
        """Run independent annealing chains in parallel on the selected GPU."""
        start = time.perf_counter()
        cp, device_name = require_cupy(device_index)
        weights = to_cupy_csr(graph, cp)
        rng = cp.random.RandomState(seed)
        n = graph.n_variables
        replicas = self.gpu_replicas
        solutions = rng.choice(cp.asarray([-1, 1], dtype=cp.int8), size=(replicas, n))
        replica_indices = cp.arange(replicas)
        total_weight = float(sum(weight for _, _, weight in graph.iter_edges()))
        initial_neighbors = weights @ solutions.T
        current_values = 0.5 * total_weight - 0.25 * cp.sum(
            solutions.T * initial_neighbors, axis=0
        )
        best_values = current_values.copy()
        best_solutions = solutions.copy()
        completed_steps = 0
        stream = cp.cuda.get_current_stream()

        for step in range(self.steps):
            if step % 32 == 0:
                stream.synchronize()
                if time_limit_seconds is not None and time.perf_counter() - start >= time_limit_seconds:
                    break

            fraction = step / max(1, self.steps - 1)
            temperature = self.initial_temperature * (
                self.final_temperature / self.initial_temperature
            ) ** fraction
            nodes = rng.randint(0, n, size=replicas)
            neighbor_sums = weights @ solutions.T
            deltas = solutions[replica_indices, nodes] * neighbor_sums[nodes, replica_indices]
            accept = (deltas >= 0) | (
                rng.random_sample(replicas) < cp.exp(deltas / max(temperature, 1e-12))
            )
            accepted_replicas = replica_indices[accept]
            accepted_nodes = nodes[accept]
            solutions[accepted_replicas, accepted_nodes] *= -1
            current_values += cp.where(accept, deltas, 0.0)
            improved = current_values > best_values
            best_solutions[improved] = solutions[improved]
            best_values[improved] = current_values[improved]
            completed_steps += 1

        stream.synchronize()
        best_index = int(cp.argmax(best_values).item())
        best = cp.asnumpy(best_solutions[best_index]).astype(np.int8)
        elapsed = time.perf_counter() - start
        status = "completed" if completed_steps == self.steps else "time_limit"
        return SolveResult(
            self.name,
            "gpu",
            best,
            float(graph.evaluate(best)),
            elapsed,
            status,
            seed,
            {
                "steps_requested": self.steps,
                "steps_completed": completed_steps,
                "execution_strategy": "parallel_independent_annealing_chains",
                "gpu_replicas": replicas,
                "device_index": device_index,
                "device": device_name,
            },
        )
