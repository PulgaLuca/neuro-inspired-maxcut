import sys
from types import SimpleNamespace

import numpy as np

from algorithms.ibm_qaoa import IBMQAOAMaxCut, _score_counts
from backends.ibm_quantum import quantum_available, resolve_ibm_token
from problems.maxcut import MaxCutProblem


def test_token_resolution_prefers_environment_and_keeps_file_secret_local(tmp_path):
    token_file = tmp_path / "apikey.json"
    token_file.write_text('{"apikey": "file-token"}', encoding="utf-8")
    assert resolve_ibm_token(
        env={"IBM_QUANTUM_TOKEN": "environment-token"}, token_file=token_file
    ) == "environment-token"
    assert resolve_ibm_token(env={}, token_file=token_file) == "file-token"
    assert resolve_ibm_token(env={}, token_file=tmp_path / "missing.json") is None


def test_quantum_availability_explains_missing_sdk(monkeypatch):
    monkeypatch.setattr("backends.ibm_quantum.importlib.util.find_spec", lambda name: None)
    available, reason = quantum_available()
    assert not available
    assert "SDK" in reason


def test_sampled_qaoa_counts_are_decoded_and_objective_recomputed():
    graph = MaxCutProblem(np.asarray([[0.0, 2.0], [2.0, 0.0]]))
    expected, solution, best = _score_counts({"01": 3, "00": 1}, graph)
    assert expected == 1.5
    assert best == 2.0
    assert graph.evaluate(solution) == 2.0


def test_qaoa_ansatz_builds_and_transpiles_with_installed_qiskit():
    from qiskit import QuantumCircuit, transpile

    from algorithms.ibm_qaoa import _qaoa_circuit

    circuit = _qaoa_circuit(
        QuantumCircuit, 3, [(0, 1, 1.0), (1, 2, 0.5)], np.asarray([0.7, 0.3]), 1
    )
    isa = transpile(
        circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1
    )
    assert isa.num_qubits == 3
    assert isa.num_clbits == 3
    assert isa.count_ops().get("measure", 0) == 3


def test_ibm_provider_submission_is_explicit_and_saves_device_metadata(monkeypatch):
    graph = MaxCutProblem(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    jobs = []

    class FakeCircuit:
        def __init__(self, qubits):
            self.qubits = qubits

        def h(self, qubits):
            pass

        def rzz(self, angle, first, second):
            pass

        def rx(self, angle, qubit):
            pass

        def measure_all(self):
            pass

    class FakeBackend:
        name = "ibm_fake_test_qpu"
        num_qubits = 5
        simulator = False

    class FakeService:
        def __init__(self, **kwargs):
            assert kwargs["token"] == "test-only-token"

        def least_busy(self, **kwargs):
            assert kwargs["simulator"] is False
            return FakeBackend()

    class FakeJob:
        def job_id(self):
            return f"fake-job-{len(jobs)}"

        def result(self, timeout=None):
            return [
                SimpleNamespace(
                    data=SimpleNamespace(
                        meas=SimpleNamespace(get_counts=lambda: {"01": 8})
                    )
                )
                for _ in jobs[-1]
            ]

    class FakeSampler:
        def __init__(self, mode):
            assert mode.name == "ibm_fake_test_qpu"

        def run(self, circuits, shots):
            jobs.append(circuits)
            return FakeJob()

    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-only-token")
    monkeypatch.setitem(
        sys.modules,
        "qiskit",
        SimpleNamespace(QuantumCircuit=FakeCircuit, transpile=lambda circuits, **kwargs: circuits),
    )
    monkeypatch.setitem(
        sys.modules,
        "qiskit_ibm_runtime",
        SimpleNamespace(QiskitRuntimeService=FakeService, SamplerV2=FakeSampler),
    )
    algorithm = IBMQAOAMaxCut(depth=1, optimizer_iterations=2, shots=8)
    result = algorithm.solve(graph, hardware="quantum", seed=42, time_limit_seconds=10)
    assert result.status == "completed"
    assert result.objective_value == graph.evaluate(result.solution) == 1.0
    assert result.metadata["provider"] == "qiskit_ibm_runtime"
    assert result.metadata["device"] == "ibm_fake_test_qpu"
    assert result.metadata["quantum_hardware_used"] is True
    assert result.metadata["job_ids"]
    assert "test-only-token" not in str(result.to_dict())
    assert len(jobs) == 3
