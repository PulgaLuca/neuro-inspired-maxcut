"""QAOA implementation that submits circuits to a real IBM Quantum QPU."""
from __future__ import annotations

import time
import os
from typing import Any

import numpy as np

from algorithms.base import Algorithm
from backends.ibm_quantum import resolve_ibm_token
from contracts import SolveResult
from problems.maxcut import MaxCutProblem


class IBMQAOAMaxCut(Algorithm):
    """Run sampled QAOA circuits through IBM Qiskit Runtime SamplerV2."""

    name = "qaoa"

    def __init__(
        self,
        depth: int = 1,
        optimizer_iterations: int = 8,
        optimizer_restarts: int = 1,
        shots: int = 1024,
        max_nodes: int = 20,
    ) -> None:
        for key, value in (
            ("depth", depth),
            ("optimizer_iterations", optimizer_iterations),
            ("optimizer_restarts", optimizer_restarts),
            ("shots", shots),
            ("max_nodes", max_nodes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be an integer >= 1.")
        self.depth = depth
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_restarts = optimizer_restarts
        self.shots = shots
        self.max_nodes = max_nodes

    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        *,
        backend_name: str | None = None,
        channel: str = "ibm_quantum_platform",
        instance: str | None = None,
        token_env_var: str = "IBM_QUANTUM_TOKEN",
        **params: Any,
    ) -> SolveResult:
        if hardware != "quantum":
            raise ValueError("IBM QAOA provider supports only the 'quantum' backend.")
        if graph.n_variables > self.max_nodes:
            raise ValueError(
                f"QAOA supports at most {self.max_nodes} nodes; "
                f"received {graph.n_variables}."
            )
        token = os.environ.get(token_env_var, "").strip() or resolve_ibm_token()
        if token is None:
            raise RuntimeError(
                "IBM Quantum token is unavailable; set IBM_QUANTUM_TOKEN or "
                "IBM_QUANTUM_TOKEN_FILE."
            )

        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except ImportError as exc:
            raise RuntimeError(
                "IBM Qiskit dependencies are missing; install neuro-inspired-maxcut[quantum]."
            ) from exc

        start = time.perf_counter()
        try:
            service_args: dict[str, Any] = {"channel": channel, "token": token}
            if instance:
                service_args["instance"] = instance
            service = QiskitRuntimeService(**service_args)
            qubits = graph.n_variables
            backend = (
                service.backend(backend_name)
                if backend_name
                else service.least_busy(
                    min_num_qubits=qubits, operational=True, simulator=False
                )
            )
            backend_label = str(backend.name)
            if getattr(backend, "num_qubits", qubits) < qubits:
                raise ValueError(
                    f"IBM backend {backend_label!r} has fewer than {qubits} qubits."
                )
            if _is_simulator(backend):
                raise ValueError(
                    f"IBM backend {backend_label!r} is a simulator; a physical QPU is required."
                )

            edges = list(graph.iter_edges())
            rng = np.random.default_rng(seed)
            best_solution = rng.choice(np.array([-1, 1], dtype=np.int8), size=qubits)
            best_value = float(graph.evaluate(best_solution))
            angles = rng.uniform(0.0, 2 * np.pi, size=2 * self.depth)
            step = np.pi / 2
            job_ids: list[str] = []
            circuit_evaluations = 0
            completed_jobs = 0
            timed_out = False

            for _restart in range(self.optimizer_restarts):
                current_circuits = [_qaoa_circuit(QuantumCircuit, qubits, edges, angles, self.depth)]
                sampled, job_id = _sample_batch(
                    current_circuits, backend, service, SamplerV2, transpile,
                    self.shots, start, time_limit_seconds,
                )
                circuit_evaluations += 1
                if job_id:
                    job_ids.append(job_id)
                if sampled is None:
                    timed_out = True
                    break
                current_energy, candidate, candidate_value = _score_counts(sampled[0], graph)
                completed_jobs += 1
                if candidate is not None and candidate_value > best_value:
                    best_solution, best_value = candidate, candidate_value

                for _iteration in range(self.optimizer_iterations):
                    if (
                        time_limit_seconds is not None
                        and time.perf_counter() - start >= time_limit_seconds
                    ):
                        timed_out = True
                        break
                    candidates = []
                    for coordinate in range(len(angles)):
                        for direction in (-1, 1):
                            candidate_angles = angles.copy()
                            candidate_angles[coordinate] = (
                                candidate_angles[coordinate] + direction * step
                            ) % (2 * np.pi)
                            candidates.append(candidate_angles)
                    circuits = [
                        _qaoa_circuit(QuantumCircuit, qubits, edges, candidate, self.depth)
                        for candidate in candidates
                    ]
                    batch, job_id = _sample_batch(
                        circuits, backend, service, SamplerV2, transpile,
                        self.shots, start, time_limit_seconds,
                    )
                    if job_id:
                        job_ids.append(job_id)
                    if batch is None:
                        timed_out = True
                        break
                    circuit_evaluations += len(circuits)
                    completed_jobs += 1
                    scores = [_score_counts(counts, graph) for counts in batch]
                    winner = int(np.argmax([score[0] for score in scores]))
                    winner_energy, winner_solution, winner_value = scores[winner]
                    if winner_solution is not None and winner_value > best_value:
                        best_solution, best_value = winner_solution, winner_value
                    if winner_energy > current_energy:
                        angles = candidates[winner]
                        current_energy = winner_energy
                    else:
                        step *= 0.5
                        if step < 1e-3:
                            break
                if timed_out:
                    break

            elapsed = time.perf_counter() - start
            objective = float(graph.evaluate(best_solution))
            return SolveResult(
                self.name,
                hardware,
                np.asarray(best_solution, dtype=np.int8),
                objective,
                elapsed,
                "time_limit" if timed_out else "completed",
                seed,
                {
                    "ansatz": "qaoa",
                    "provider": "qiskit_ibm_runtime",
                    "device": backend_label,
                    "quantum_hardware_used": completed_jobs > 0,
                    "channel": channel,
                    "instance": instance,
                    "depth": self.depth,
                    "shots_per_circuit": self.shots,
                    "optimizer": "seeded_coordinate_search",
                    "optimizer_iterations": self.optimizer_iterations,
                    "optimizer_restarts": self.optimizer_restarts,
                    "circuit_evaluations": circuit_evaluations,
                    "completed_jobs": completed_jobs,
                    "job_ids": job_ids,
                    "best_observed_cut": objective,
                    "sampling_is_stochastic": True,
                },
            )
        except Exception as exc:
            message = str(exc).replace(token, "[REDACTED]")
            raise RuntimeError(
                f"IBM Quantum QAOA execution failed ({type(exc).__name__}): {message}"
            ) from None
        finally:
            token = ""  # Do not retain credentials on the algorithm instance or result.


def _qaoa_circuit(
    circuit_type: Any,
    n: int,
    edges: list[tuple[int, int, float]],
    angles: np.ndarray,
    depth: int,
) -> Any:
    circuit = circuit_type(n)
    circuit.h(range(n))
    for layer in range(depth):
        gamma, beta = angles[2 * layer], angles[2 * layer + 1]
        for u, v, weight in edges:
            circuit.rzz(float(gamma * weight), u, v)
        for qubit in range(n):
            circuit.rx(float(2 * beta), qubit)
    circuit.measure_all()
    return circuit


def _is_simulator(backend: Any) -> bool:
    simulator = getattr(backend, "simulator", None)
    if simulator is not None:
        return bool(simulator)
    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        configuration = configuration()
    return bool(getattr(configuration, "simulator", False))


def _sample_batch(
    circuits: list[Any],
    backend: Any,
    service: Any,
    sampler_type: Any,
    transpile: Any,
    shots: int,
    started: float,
    time_limit_seconds: float | None,
) -> tuple[list[dict[str, int]] | None, str | None]:
    if (
        time_limit_seconds is not None
        and time.perf_counter() - started >= time_limit_seconds
    ):
        return None, None
    isa_circuits = transpile(circuits, backend=backend, optimization_level=1)
    sampler = sampler_type(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    job_id = str(job.job_id())
    timeout = None
    if time_limit_seconds is not None:
        timeout = max(0.001, time_limit_seconds - (time.perf_counter() - started))
    try:
        result = job.result(timeout=timeout)
    except Exception as exc:
        if not isinstance(exc, TimeoutError) and type(exc).__name__ not in {
            "JobTimeoutError", "RuntimeJobTimeoutError", "RuntimeJobMaxTimeoutError"
        }:
            raise
        try:
            job.cancel()
        except Exception:
            pass
        return None, job_id
    batches = []
    for pub_result in result:
        data = pub_result.data
        register = getattr(data, "meas", None)
        if register is None:
            names = [name for name in data.keys()]
            if len(names) != 1:
                raise RuntimeError("IBM Sampler result has no unique measurement register.")
            register = data[names[0]]
        batches.append(register.get_counts())
    return batches, job_id


def _score_counts(
    counts: dict[str, int], graph: MaxCutProblem
) -> tuple[float, np.ndarray | None, float]:
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("IBM Sampler returned an empty measurement distribution.")
    expected_cut = 0.0
    best_cut = float("-inf")
    best_solution = None
    for raw_bits, count in counts.items():
        bits = raw_bits.replace(" ", "")
        if len(bits) != graph.n_variables or set(bits) - {"0", "1"}:
            raise RuntimeError("IBM Sampler returned a bitstring with an invalid width.")
        solution = np.asarray(
            [1 if bits[-1 - qubit] == "0" else -1 for qubit in range(graph.n_variables)],
            dtype=np.int8,
        )
        value = float(graph.evaluate(solution))
        expected_cut += value * count / total
        if value > best_cut:
            best_cut, best_solution = value, solution
    return expected_cut, best_solution, best_cut
