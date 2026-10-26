from pathlib import Path

import numpy as np
import json
import pytest

from algorithms.registry import (
    create_algorithm,
    register_algorithm,
    register_backend_implementation,
    supported_backends,
)
from backends.registry import (
    BackendCapability,
    get_backend,
    register_backend,
)
from config.loader import load_config
from config.validation import validate_config
from experiments.runner import derive_run_seed, execute_plan, plan_runs
from graphs.erdos_renyi import erdos_renyi
from metrics.physical_energy import (
    PhysicalEnergyMeter,
    register_physical_energy_provider,
)
from problems.maxcut import MaxCutProblem

ROOT = Path(__file__).resolve().parents[1]


def test_example_configurations_validate_and_plan_cpu_runs():
    for name in (
        "benchmark_local_search_cpu.yaml",
        "benchmark_heuristics_cpu.yaml",
        "benchmark_gset.yaml",
    ):
        config = load_config(ROOT / "configs" / name)
        assert validate_config(config) == []
        plan = plan_runs(config)
        assert plan
        assert all(item["supported"] for item in plan)

    gpu_config = load_config(ROOT / "configs" / "benchmark_cpu_gpu.yaml")
    assert validate_config(gpu_config) == []
    gpu_plan = plan_runs(gpu_config)
    assert any(item["backend"]["name"] == "gpu" for item in gpu_plan)


def test_registry_declares_only_real_cpu_implementations():
    assert supported_backends("simulated_annealing") == frozenset({"cpu", "gpu"})
    assert supported_backends("exact") == frozenset({"cpu", "gpu"})
    assert supported_backends("local_search") == frozenset({"cpu", "gpu"})
    assert supported_backends("hopfield") == frozenset({"cpu"})
    assert supported_backends("boltzmann") == frozenset({"cpu"})
    assert supported_backends("spiking_neural_network") == frozenset({"cpu"})
    assert supported_backends("qaoa") == frozenset({"quantum_simulator", "quantum"})
    assert create_algorithm("local_search", {"max_iterations": 2}).name == "local_search"
    assert create_algorithm("hopfield", {"max_iterations": 2}).name == "hopfield"
    assert create_algorithm("boltzmann", {"steps": 2}).name == "boltzmann"
    assert create_algorithm("spiking_neural_network", {"time_steps": 2}).name == "spiking_neural_network"
    assert get_backend("cpu").available
    assert get_backend("quantum_simulator").available
    quantum_backend = get_backend("quantum")
    assert quantum_backend is not None
    if not quantum_backend.available:
        assert quantum_backend.reason
    assert not get_backend("gpu").available


def test_run_seed_is_shared_across_algorithm_and_backend_comparisons():
    assert derive_run_seed(10, "G1", 2) == 2197
    assert derive_run_seed(10, "G1", 2) != derive_run_seed(10, "G1", 3)
    assert derive_run_seed(10, "G1", 2) != derive_run_seed(10, "G2", 2)


def test_plan_marks_unavailable_hardware_without_claiming_support():
    config = load_config(ROOT / "configs" / "benchmark_local_search_cpu.yaml")
    config["backends"] = [{"name": "gpu", "params": {}}]
    plan = plan_runs(config)
    assert plan
    if get_backend("gpu").available:
        assert plan[0]["supported"]
    else:
        assert not plan[0]["supported"]
        assert "CuPy" in plan[0]["compatibility"] or "CUDA" in plan[0]["compatibility"]


def test_gpu_solve_never_falls_back_to_cpu_when_provider_is_missing():
    if get_backend("gpu").available:
        pytest.skip("A CUDA GPU is available; this test covers the missing-provider case.")
    graph = erdos_renyi(4, 0.5, seed=3)
    algorithms = [
        create_algorithm("exact", {"max_nodes": 8}),
        create_algorithm("local_search", {"max_iterations": 5}),
        create_algorithm("simulated_annealing", {"steps": 5}),
    ]
    for algorithm in algorithms:
        with pytest.raises(RuntimeError, match="CuPy|CUDA|GPU"):
            algorithm.solve(graph, hardware="gpu", seed=1)


def test_erdos_renyi_and_algorithm_are_seed_reproducible_and_verified():
    first = erdos_renyi(12, 0.3, seed=7, weighted=True)
    second = erdos_renyi(12, 0.3, seed=7, weighted=True)
    assert np.array_equal(first.weights.toarray(), second.weights.toarray())
    a = create_algorithm("simulated_annealing", {"steps": 100})
    result1 = a.solve(first, hardware="cpu", seed=19)
    result2 = a.solve(second, hardware="cpu", seed=19)
    assert np.array_equal(result1.solution, result2.solution)
    assert np.isclose(result1.objective_value, first.evaluate(result1.solution))


def test_exact_cpu_returns_verified_optimum():
    graph = erdos_renyi(7, 0.4, seed=12, weighted=True)
    result = create_algorithm("exact", {"max_nodes": 10}).solve(
        graph, hardware="cpu", seed=2
    )
    brute_force = max(
        graph.evaluate(np.asarray(
            [1 if mask & (1 << bit) else -1 for bit in range(7)], dtype=np.int8
        ))
        for mask in range(1 << 7)
    )
    assert result.status == "completed"
    assert result.objective_value == pytest.approx(brute_force)


def test_hopfield_is_reproducible_and_converges_to_a_one_flip_stable_solution():
    graph = erdos_renyi(16, 0.3, seed=21, weighted=True)
    algorithm = create_algorithm("hopfield", {"max_iterations": 10_000})
    first = algorithm.solve(graph, hardware="cpu", seed=7)
    second = algorithm.solve(graph, hardware="cpu", seed=7)
    assert first.status == "completed"
    assert first.metadata["converged"]
    assert first.metadata["dynamics"] == "asynchronous_random_order"
    assert np.array_equal(first.solution, second.solution)
    assert first.objective_value == pytest.approx(graph.evaluate(first.solution))
    assert all(graph.flip_delta(first.solution, node) <= 0 for node in range(graph.n_variables))


def test_hopfield_supports_signed_gset_style_weights():
    weights = np.array(
        [[0.0, -2.0, 3.0], [-2.0, 0.0, 1.0], [3.0, 1.0, 0.0]]
    )
    graph = MaxCutProblem(weights)
    result = create_algorithm("hopfield").solve(graph, hardware="cpu", seed=9)
    assert result.metadata["converged"]
    assert result.objective_value == pytest.approx(graph.evaluate(result.solution))
    assert all(graph.flip_delta(result.solution, node) <= 0 for node in range(3))


def test_hopfield_reports_time_limit_and_rejects_gpu_explicitly():
    graph = erdos_renyi(8, 0.5, seed=5)
    algorithm = create_algorithm("hopfield")
    timed = algorithm.solve(graph, hardware="cpu", seed=1, time_limit_seconds=0)
    assert timed.status == "time_limit"
    assert np.isfinite(timed.objective_value)
    with pytest.raises(ValueError, match="Hardware not supported"):
        algorithm.solve(graph, hardware="gpu", seed=1)


def test_boltzmann_gibbs_sampling_is_reproducible_and_verifies_best_state():
    graph = erdos_renyi(12, 0.35, seed=31, weighted=True)
    algorithm = create_algorithm("boltzmann", {"steps": 500, "temperature": 0.8})
    first = algorithm.solve(graph, hardware="cpu", seed=11)
    second = algorithm.solve(graph, hardware="cpu", seed=11)
    assert first.status == "completed"
    assert first.metadata["dynamics"] == "asynchronous_single_spin_gibbs"
    assert first.metadata["gibbs_updates"] == 500
    assert np.array_equal(first.solution, second.solution)
    assert first.objective_value == pytest.approx(graph.evaluate(first.solution))
    assert first.objective_value >= first.metadata["final_chain_objective_value"]
    with pytest.raises(ValueError, match="Hardware not supported"):
        algorithm.solve(graph, hardware="gpu", seed=11)
    signed_graph = MaxCutProblem(
        np.array([[0.0, -2.0, 3.0], [-2.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    )
    signed_result = algorithm.solve(signed_graph, hardware="cpu", seed=11)
    assert signed_result.objective_value == pytest.approx(
        signed_graph.evaluate(signed_result.solution)
    )


def test_boltzmann_gpu_combination_is_marked_unsupported():
    config = load_config(ROOT / "configs" / "benchmark_heuristics_cpu.yaml")
    config["algorithms"] = [{"name": "boltzmann", "params": {"steps": 2}}]
    config["backends"] = [{"name": "gpu", "params": {}}]
    plan = plan_runs(config)
    assert plan and all(not item["supported"] for item in plan)
    assert all(
        "boltzmann has no gpu implementation" in item["compatibility"]
        for item in plan
    )


def test_spiking_network_is_reproducible_and_only_reports_cpu_simulation():
    graph = erdos_renyi(10, 0.4, seed=17, weighted=True)
    algorithm = create_algorithm(
        "spiking_neural_network",
        {"time_steps": 100, "threshold": 0.5, "synaptic_scale": 0.25},
    )
    first = algorithm.solve(graph, hardware="cpu", seed=23)
    second = algorithm.solve(graph, hardware="cpu", seed=23)
    assert first.status == "completed"
    assert first.metadata["model"] == "paired_competitive_lif"
    assert first.metadata["neuromorphic_hardware_used"] is False
    assert first.metadata["provider"] == "cpu_snn_simulator"
    assert np.array_equal(first.solution, second.solution)
    assert first.objective_value == pytest.approx(graph.evaluate(first.solution))
    with pytest.raises(ValueError, match="Hardware not supported"):
        algorithm.solve(graph, hardware="neuromorphic", seed=23)


def test_neuromorphic_snn_plan_is_explicitly_unavailable_without_provider():
    config = load_config(ROOT / "configs" / "benchmark_heuristics_cpu.yaml")
    config["backends"] = [{"name": "neuromorphic", "params": {}}]
    plan = plan_runs(config)
    snn_runs = [
        item for item in plan
        if item["algorithm"]["name"] == "spiking_neural_network"
    ]
    assert snn_runs
    assert all(not item["supported"] for item in snn_runs)
    assert all("provider" in item["compatibility"] for item in snn_runs)


def test_qaoa_simulator_is_reproducible_and_returns_verified_cut():
    graph = erdos_renyi(5, 0.5, seed=11, weighted=True)
    algorithm = create_algorithm(
        "qaoa", {"depth": 1, "optimizer_iterations": 8, "max_nodes": 8},
        hardware="quantum_simulator",
    )
    first = algorithm.solve(graph, hardware="quantum_simulator", seed=19)
    second = algorithm.solve(graph, hardware="quantum_simulator", seed=19)
    assert first.status == "completed"
    assert np.array_equal(first.solution, second.solution)
    assert first.objective_value == pytest.approx(graph.evaluate(first.solution))
    assert first.metadata["provider"] == "numpy_statevector_simulator"
    assert first.metadata["quantum_hardware_used"] is False
    assert first.metadata["statevector_dimension"] == 32
    with pytest.raises(ValueError, match="quantum_simulator"):
        algorithm.solve(graph, hardware="quantum", seed=19)


def test_qaoa_configuration_accepts_configured_ibm_backend():
    config = {
        "experiment": {"seed": 1, "repetitions": 1, "output": {"root": "results"}},
        "execution": {"time_limit_seconds": 2},
        "graphs": [{"name": "tiny", "type": "erdos_renyi", "sizes": [5]}],
        "algorithms": [{"name": "qaoa", "params": {"max_nodes": 4}}],
        "backends": [{"name": "quantum", "params": {}}],
    }
    assert validate_config(config) == []


def test_qaoa_parameters_are_validated():
    config = {
        "experiment": {"seed": 1, "repetitions": 1, "output": {"root": "results"}},
        "execution": {"time_limit_seconds": 2},
        "graphs": [{"name": "tiny", "type": "erdos_renyi", "sizes": [5]}],
        "algorithms": [{"name": "qaoa", "params": {"depth": 0}}],
        "backends": [{"name": "quantum_simulator", "params": {}}],
    }
    assert any("params.depth" in error for error in validate_config(config))


def test_backend_specific_algorithm_provider_can_be_registered():
    def marker_factory(params):
        return "cpu", params

    def hardware_factory(params):
        return "neuromorphic", params

    register_algorithm("test_provider_extension", marker_factory, backends={"cpu"})
    register_backend_implementation(
        "test_provider_extension", "test_neuro_target", hardware_factory
    )
    register_backend(BackendCapability("test_neuro_target", True))
    assert get_backend("test_neuro_target").available
    assert create_algorithm(
        "test_provider_extension", {"x": 1}, hardware="test_neuro_target"
    ) == ("neuromorphic", {"x": 1})


def test_neuromorphic_energy_requires_an_explicit_meter_provider():
    unavailable = PhysicalEnergyMeter.start("neuromorphic").finish()
    assert unavailable["joules"] is None
    assert unavailable["status"] == "unavailable"

    class Counter:
        def read(self):
            return (0,)

        def difference_joules(self, start, end):
            return 2.75

        def close(self):
            pass

    register_physical_energy_provider(
        "test_neuromorphic_meter",
        lambda params: PhysicalEnergyMeter("test_counter", "device", Counter(), (0,)),
    )
    measured = PhysicalEnergyMeter.start("test_neuromorphic_meter").finish()
    assert measured["joules"] == pytest.approx(2.75)
    assert measured["status"] == "measured"


def test_quantum_simulator_energy_uses_cpu_package_meter(monkeypatch):
    class Counter:
        def read(self):
            return (0,)

        def difference_joules(self, start, end):
            return 1.25

        def close(self):
            pass

    monkeypatch.setattr(
        "metrics.physical_energy._rapl_package_counter", lambda: Counter()
    )
    measurement = PhysicalEnergyMeter.start("quantum_simulator").finish()
    assert measurement["joules"] == pytest.approx(1.25)
    assert measurement["status"] == "measured"
    assert measurement["scope"] == "cpu_package"


def test_solution_validation_rejects_invalid_spins():
    graph = erdos_renyi(3, 0.5, seed=1)
    try:
        graph.evaluate(np.array([1, 0, -1]))
    except ValueError:
        return
    raise AssertionError("invalid spin assignment was accepted")


def test_end_to_end_run_persists_verified_result(tmp_path):
    config = {
        "experiment": {"name": "test", "seed": 5, "repetitions": 1,
                       "output": {"root": str(tmp_path), "save_graph_image": False,
                                  "save_solution": True, "save_logs": True}},
        "execution": {"time_limit_seconds": 2, "fail_fast": False},
        "graphs": [{"name": "small", "type": "erdos_renyi", "sizes": [8],
                    "density": 0.4, "weighted": True}],
        "algorithms": [
            {"name": "local_search", "params": {"max_iterations": 30}},
            {"name": "hopfield", "params": {"max_iterations": 1000}},
            {"name": "boltzmann", "params": {"steps": 100, "temperature": 1.0}},
            {"name": "spiking_neural_network", "params": {"time_steps": 100}},
        ],
        "backends": [{"name": "cpu", "params": {}}],
    }
    plan = plan_runs(config)
    counts = execute_plan(config, Path("unused.yaml"), tmp_path, plan)
    assert counts["completed"] == 4
    record = json.loads((tmp_path / "runs/run_000001/result.json").read_text())
    hopfield_record = json.loads((tmp_path / "runs/run_000002/result.json").read_text())
    boltzmann_record = json.loads((tmp_path / "runs/run_000003/result.json").read_text())
    snn_record = json.loads((tmp_path / "runs/run_000004/result.json").read_text())
    assert record["schema_version"] == 1
    graph_data = json.loads((tmp_path / "runs/run_000001/graph.json").read_text())
    assert record["objective_value"] == pytest.approx(
        sum(edge["weight"] for edge in graph_data["edges"]
            if record["solution"][edge["u"]] != record["solution"][edge["v"]])
    )
    assert hopfield_record["algorithm"] == "hopfield"
    assert hopfield_record["metadata"]["converged"]
    assert hopfield_record["seed"] == record["seed"]
    assert boltzmann_record["algorithm"] == "boltzmann"
    assert boltzmann_record["metadata"]["gibbs_updates"] == 100
    assert boltzmann_record["seed"] == record["seed"]
    assert snn_record["algorithm"] == "spiking_neural_network"
    assert snn_record["metadata"]["neuromorphic_hardware_used"] is False
    assert snn_record["seed"] == record["seed"]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 4


def test_unsupported_backend_is_persisted_as_failure(tmp_path):
    if get_backend("gpu").available:
        pytest.skip("A CUDA GPU is available; this test covers unavailable-provider reporting.")
    config = {
        "experiment": {"seed": 1, "repetitions": 1,
                       "output": {"save_graph_image": False, "save_solution": False,
                                  "save_logs": False}},
        "execution": {"time_limit_seconds": 1, "fail_fast": False},
        "graphs": [{"name": "small", "type": "erdos_renyi", "sizes": [3]}],
        "algorithms": [{"name": "local_search", "params": {}}],
        "backends": [{"name": "gpu", "params": {}}],
    }
    counts = execute_plan(config, Path("unused.yaml"), tmp_path, plan_runs(config))
    record = json.loads((tmp_path / "runs/run_000001/result.json").read_text())
    assert counts["failed"] == 1
    assert record["status"] == "failed"
    assert record["solution"] is None and record["objective_value"] is None
    assert "CuPy" in record["error"] or "CUDA" in record["error"]


def test_validate_reports_unknown_algorithm():
    config = load_config(ROOT / "configs" / "benchmark_local_search_cpu.yaml")
    config["algorithms"] = [{"name": "unknown_algorithm", "params": {}}]
    assert any("not implemented" in error for error in validate_config(config))


def test_hopfield_parameters_are_validated():
    config = load_config(ROOT / "configs" / "benchmark_heuristics_cpu.yaml")
    config["algorithms"][1]["params"]["max_iterations"] = 0
    assert any("algorithms[1].params.max_iterations" in error
               for error in validate_config(config))


def test_boltzmann_parameters_are_validated():
    config = load_config(ROOT / "configs" / "benchmark_heuristics_cpu.yaml")
    config["algorithms"][2]["params"]["temperature"] = 0
    assert any("algorithms[2].params.temperature" in error
               for error in validate_config(config))


def test_snn_parameters_are_validated():
    config = load_config(ROOT / "configs" / "benchmark_heuristics_cpu.yaml")
    config["algorithms"][3]["params"]["threshold"] = 0
    assert any("algorithms[3].params.threshold" in error
               for error in validate_config(config))


def test_empty_plan_sections_are_rejected():
    config = load_config(ROOT / "configs" / "benchmark_local_search_cpu.yaml")
    config["backends"] = []
    assert any("backends" in error and "non-empty" in error
               for error in validate_config(config))
