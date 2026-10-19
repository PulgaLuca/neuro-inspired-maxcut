import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from config.loader import load_gset
from config.validation import validate_config
from experiments.runner import execute_plan, plan_runs
from problems.maxcut import MaxCutProblem
from storage.results import append_result


def _write_instance(root: Path, name: str, contents: str) -> None:
    instance_dir = root / name
    instance_dir.mkdir(parents=True)
    (instance_dir / f"{name}.txt").write_text(contents, encoding="utf-8")


def test_load_gset_instances_as_sparse_signed_graphs(tmp_path: Path) -> None:
    _write_instance(tmp_path, "G1", "3 3\n1 2 2\n2 3 -1\n1 3 0.5\n")
    _write_instance(tmp_path, "G2", "2 1\n1 2 4\n")

    graphs = load_gset(["G1", "G2"], tmp_path)
    graph = graphs["G1"]

    assert set(graphs) == {"G1", "G2"}
    assert sparse.issparse(graph.weights)
    assert graph.n_variables == 3
    assert graph.n_edges == 3
    assert graph.evaluate(np.array([1, -1, 1])) == pytest.approx(1)
    assert list(graph.iter_edges()) == [(0, 1, 2.0), (0, 2, 0.5), (1, 2, -1.0)]


def test_sparse_flip_delta_matches_objective_difference() -> None:
    graph = MaxCutProblem(
        sparse.csr_matrix(
            np.array([[0, 2, -1], [2, 0, 3], [-1, 3, 0]], dtype=float)
        )
    )
    solution = np.array([1, -1, 1], dtype=np.int8)

    for node in range(graph.n_variables):
        flipped = solution.copy()
        flipped[node] *= -1
        assert graph.flip_delta(solution, node) == pytest.approx(
            graph.evaluate(flipped) - graph.evaluate(solution)
        )


def test_validate_config_checks_gset_files_and_plans_each_instance(tmp_path: Path) -> None:
    _write_instance(tmp_path, "G1", "2 1\n1 2 1\n")
    _write_instance(tmp_path, "G2", "3 2\n1 2 1\n2 3 1\n")
    config = {
        "experiment": {"repetitions": 1, "output": {"root": str(tmp_path / "results")}},
        "execution": {},
        "graphs": [{"name": "gset", "type": "gset", "instances": "all", "path": str(tmp_path)}],
        "algorithms": [{"name": "simulated_annealing", "params": {"steps": 2}}],
        "backends": [{"name": "cpu", "params": {}}],
    }

    assert validate_config(config) == []
    planned = plan_runs(config)
    assert [item["instance"] for item in planned] == ["G1", "G2"]
    assert all(item["size"] is None for item in planned)


def test_runner_saves_gset_results_in_standard_format(tmp_path: Path) -> None:
    _write_instance(tmp_path, "G1", "3 3\n1 2 2\n2 3 -1\n1 3 0.5\n")
    experiment_dir = tmp_path / "experiment"
    (experiment_dir / "runs").mkdir(parents=True)
    (experiment_dir / "results.jsonl").touch()
    (experiment_dir / "summary.csv").touch()
    config = {
        "experiment": {
            "seed": 9,
            "repetitions": 1,
            "output": {
                "root": str(tmp_path / "results"),
                "save_graph_image": False,
                "save_solution": True,
                "save_logs": False,
            },
        },
        "execution": {"time_limit_seconds": 5, "fail_fast": True},
        "graphs": [
            {"name": "gset", "type": "gset", "instances": ["G1"], "path": str(tmp_path)}
        ],
        "algorithms": [{"name": "simulated_annealing", "params": {"steps": 3}}],
        "backends": [{"name": "cpu", "params": {}}],
    }

    counts = execute_plan(config, Path("config.yaml"), experiment_dir, plan_runs(config))
    result = json.loads(
        (experiment_dir / "runs" / "run_000001" / "result.json").read_text(encoding="utf-8")
    )
    graph_data = json.loads(
        (experiment_dir / "runs" / "run_000001" / "graph.json").read_text(encoding="utf-8")
    )

    assert counts == {"completed": 1, "time_limit": 0, "failed": 0}
    assert result["graph_id"] == "G1"
    assert result["graph_type"] == "gset"
    assert result["instance"] == "G1"
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 3
    assert graph_data["type"] == "gset"
    assert len(graph_data["edges"]) == 3


def test_summary_csv_keeps_fields_added_by_later_graph_types(tmp_path: Path) -> None:
    (tmp_path / "summary.csv").touch()
    append_result(tmp_path, {"graph_id": "er_n3", "runtime_seconds": 1.5})
    append_result(
        tmp_path,
        {"graph_id": "G1", "runtime_seconds": 2.5, "graph_type": "gset"},
    )

    rows = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "graph_id,runtime_seconds,graph_type"
    assert rows[1] == "er_n3,1.5,"
    assert rows[2] == "G1,2.5,gset"
