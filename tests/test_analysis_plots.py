import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from analysis.plots import (
    normalized_comparison,
    pareto_frontier,
    plot_measured_energy_tradeoffs,
    plot_normalized_comparison,
    plot_pareto_frontier,
    plot_quality_by_graph_size,
    plot_scaling,
    plot_time_quality,
    prepare_runs,
)


@pytest.fixture
def sample_runs():
    return pd.DataFrame(
        [
            {"run_id": "exact1", "graph_id": "er_n10", "graph_type": "erdos_renyi", "density": 0.05,
             "n_nodes": 10, "algorithm": "exact", "backend": "cpu", "status": "completed",
             "cut_value": 90, "cut_fraction": 0.9, "runtime_seconds": 2.0,
             "physical_energy_joules": 10.0, "physical_energy_status": "measured"},
            {"run_id": "gpu1", "graph_id": "er_n10", "graph_type": "erdos_renyi", "density": 0.05,
             "n_nodes": 10, "algorithm": "local_search", "backend": "gpu", "status": "completed",
             "cut_value": 81, "cut_fraction": 0.81, "runtime_seconds": 1.0,
             "physical_energy_joules": None, "physical_energy_status": "unavailable"},
            {"run_id": "cpu1", "graph_id": "er_n10", "graph_type": "erdos_renyi", "density": 0.05,
             "n_nodes": 10, "algorithm": "local_search", "backend": "cpu", "status": "completed",
             "cut_value": 72, "cut_fraction": 0.72, "runtime_seconds": 4.0,
             "physical_energy_joules": 20.0, "physical_energy_status": "measured"},
            {"run_id": "dense1", "graph_id": "er_n50", "graph_type": "erdos_renyi", "density": 0.6,
             "n_nodes": 50, "algorithm": "local_search", "backend": "cpu", "status": "completed",
             "cut_value": 40, "cut_fraction": 0.8, "runtime_seconds": 5.0,
             "physical_energy_joules": 15.0, "physical_energy_status": "measured"},
            {"run_id": "failed", "graph_id": "er_n10", "graph_type": "erdos_renyi", "density": 0.05,
             "n_nodes": 10, "algorithm": "broken", "backend": "cpu", "status": "failed",
             "cut_value": None, "cut_fraction": None, "runtime_seconds": None,
             "physical_energy_joules": None, "physical_energy_status": "unavailable"},
        ]
    )


def test_prepare_runs_only_retains_explicit_measured_energy(sample_runs):
    prepared = prepare_runs(sample_runs)
    assert prepared.loc[prepared.run_id == "gpu1", "measured_energy_joules"].isna().all()
    assert prepared.loc[prepared.run_id == "exact1", "graph_family"].iloc[0] == "sparso"
    assert prepared.loc[prepared.run_id == "dense1", "graph_family"].iloc[0] == "denso"


def test_normalization_uses_exact_cpu_then_same_algorithm_cpu_speedup(sample_runs):
    compared = normalized_comparison(sample_runs)
    gpu = compared.loc[compared.run_id == "gpu1"].iloc[0]
    assert gpu.quality_reference == "exact CPU"
    assert gpu.quality_ratio == pytest.approx(0.9)
    assert gpu.speedup_vs_cpu == pytest.approx(4.0)


def test_pareto_uses_only_measured_energy_and_marks_nondominated_rows(sample_runs):
    pareto = pareto_frontier(sample_runs)
    assert set(pareto.run_id) == {"exact1", "cpu1", "dense1"}
    flags = pareto.set_index("run_id").pareto_optimal.to_dict()
    assert flags == {"exact1": True, "cpu1": False, "dense1": True}


def test_plot_builders_draw_and_export_figures(sample_runs, tmp_path):
    figures = [
        plot_time_quality(sample_runs, output_dir=tmp_path),
        *plot_measured_energy_tradeoffs(sample_runs, output_dir=tmp_path),
        plot_quality_by_graph_size(sample_runs, output_dir=tmp_path),
        *plot_scaling(sample_runs, output_dir=tmp_path),
    ]
    _, normalized_figure = plot_normalized_comparison(sample_runs, output_dir=tmp_path)
    _, pareto_figure = plot_pareto_frontier(sample_runs, output_dir=tmp_path)
    figures.extend([normalized_figure, pareto_figure])
    for figure in figures:
        figure.canvas.draw()
        import matplotlib.pyplot as plt

        plt.close(figure)
    assert len(list(tmp_path.glob("*.png"))) == 8
    assert len(list(tmp_path.glob("*.pdf"))) == 8


def test_empty_runs_keep_missing_energy_explicit():
    empty = pd.DataFrame(columns=[
        "graph_id", "algorithm", "backend", "status", "n_nodes", "cut_value",
        "cut_fraction", "runtime_seconds", "physical_energy_joules",
        "physical_energy_status",
    ])
    assert pareto_frontier(empty).empty
    for figure in plot_measured_energy_tradeoffs(empty):
        figure.canvas.draw()
        import matplotlib.pyplot as plt

        plt.close(figure)
