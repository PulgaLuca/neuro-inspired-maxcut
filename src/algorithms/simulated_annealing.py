import numpy as np

from contracts import SolveResult
import time

class SimulatedAnnealing:
    name = "simulated_annealing"

    def __init__(self, steps: int = 10000, initial_temperature: float = 10.0, final_temperature: float = 0.01):
        if steps < 1: 
            raise ValueError("steps deve essere >= 1.")
        if initial_temperature <= 0 or final_temperature <= 0:
            raise ValueError("Le temperature devono essere positive.")
        if final_temperature > initial_temperature:
            raise ValueError("final_temperature deve essere <= initial_temperature.")
        
        self.steps = steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

    
    def solve(
        self,
        graph,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
    ) -> SolveResult:
        if hardware == "cpu":
            return self._solve_cpu(graph, seed, time_limit_seconds)
        if hardware == "gpu":
            return self._solve_gpu()

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


    def _solve_gpu(self) -> SolveResult:
        pass
