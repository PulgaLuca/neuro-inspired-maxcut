"""Plot builders for comparing Max-Cut benchmark runs.

All functions accept the normalized run dataframe used by the analysis
notebook. Physical-energy plots use only rows explicitly marked ``measured``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GROUP = ["graph_id", "algorithm", "backend"]


def prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with run status, graph family and measured energy normalized."""
    frame = runs.copy()
    for column in ("algorithm", "backend", "graph_id", "status"):
        if column not in frame:
            frame[column] = "unknown"
    for column in (
        "graph_type", "n_nodes", "density", "runtime_seconds", "cut_value",
        "cut_fraction", "physical_energy_joules", "physical_energy_status",
    ):
        if column not in frame:
            frame[column] = np.nan
    if "graph_family" not in frame:
        frame["graph_family"] = frame.apply(_infer_graph_family, axis=1)
    else:
        frame["graph_family"] = frame["graph_family"].fillna("other")
    measured = frame["physical_energy_status"].eq("measured")
    frame["measured_energy_joules"] = pd.to_numeric(
        frame["physical_energy_joules"], errors="coerce"
    ).where(measured)
    for column in ("runtime_seconds", "cut_value", "cut_fraction", "n_nodes", "density"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _infer_graph_family(row: pd.Series) -> str:
    graph_type = str(row.get("graph_type", "")).lower()
    if graph_type == "gset":
        return "gset"
    if graph_type in {"erdos_renyi", "erdos-renyi", "er"}:
        density = row.get("density")
        if pd.isna(density):
            return "erdos_renyi (densità ignota)"
        if density < 0.1:
            return "sparso"
        if density >= 0.5:
            return "denso"
        return "densità media"
    return graph_type or "other"


def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["status"].eq("completed")].copy()


def _save(fig: plt.Figure, output_dir: str | Path | None, name: str) -> None:
    if output_dir is None:
        return
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")


def _empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_time_quality(
    runs: pd.DataFrame, *, output_dir: str | Path | None = None
) -> plt.Figure:
    """Scatter all run timings/quality and show group medians and IQR bars."""
    frame = _completed(prepare_runs(runs)).dropna(
        subset=["runtime_seconds", "cut_fraction"]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    if frame.empty:
        _empty_axis(ax, "Nessun run completato con tempo e qualità disponibili")
    else:
        for (graph_id, algorithm, backend), group in frame.groupby(_GROUP, dropna=False):
            label = f"{graph_id} · {algorithm} · {backend}"
            ax.scatter(group["runtime_seconds"], group["cut_fraction"], alpha=0.55, s=28)
            time_values = group["runtime_seconds"]
            quality_values = group["cut_fraction"]
            time_median = time_values.median()
            quality_median = quality_values.median()
            ax.errorbar(
                time_median, quality_median,
                xerr=[[time_median - time_values.quantile(0.25)],
                      [time_values.quantile(0.75) - time_median]],
                yerr=[[quality_median - quality_values.quantile(0.25)],
                      [quality_values.quantile(0.75) - quality_median]],
                fmt="o", markersize=8, capsize=3, label=label,
            )
        ax.set_xlabel("Tempo per run (s)")
        ax.set_ylabel("Qualità: frazione del peso assoluto tagliato")
        ax.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.25)
    ax.set_title("Tempo vs qualità: run e mediana [IQR]")
    fig.tight_layout()
    _save(fig, output_dir, "tempo_vs_qualita")
    return fig


def plot_measured_energy_tradeoffs(
    runs: pd.DataFrame, *, output_dir: str | Path | None = None
) -> tuple[plt.Figure, plt.Figure]:
    """Plot measured joules vs quality and measured joules vs time."""
    frame = _completed(prepare_runs(runs)).dropna(
        subset=["measured_energy_joules", "runtime_seconds", "cut_fraction"]
    )
    fig_quality, ax_quality = plt.subplots(figsize=(9, 6))
    fig_time, ax_time = plt.subplots(figsize=(9, 6))
    if frame.empty:
        message = "Nessuna run completata con misura fisica di energia valida"
        _empty_axis(ax_quality, message)
        _empty_axis(ax_time, message)
    else:
        for (graph_id, algorithm, backend), group in frame.groupby(_GROUP, dropna=False):
            label = f"{graph_id} · {algorithm} · {backend}"
            ax_quality.scatter(
                group["measured_energy_joules"], group["cut_fraction"],
                alpha=0.7, label=label,
            )
            ax_time.scatter(
                group["runtime_seconds"], group["measured_energy_joules"],
                alpha=0.7, label=label,
            )
        ax_quality.set_xlabel("Energia fisica misurata (J)")
        ax_quality.set_ylabel("Qualità: frazione del peso assoluto tagliato")
        ax_time.set_xlabel("Tempo per run (s)")
        ax_time.set_ylabel("Energia fisica misurata (J)")
        for ax in (ax_quality, ax_time):
            ax.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(alpha=0.25)
    ax_quality.set_title("Energia misurata vs qualità")
    ax_time.set_title("Tempo vs energia misurata")
    fig_quality.tight_layout()
    fig_time.tight_layout()
    _save(fig_quality, output_dir, "energia_misurata_vs_qualita")
    _save(fig_time, output_dir, "tempo_vs_energia_misurata")
    return fig_quality, fig_time


def plot_quality_by_graph_size(
    runs: pd.DataFrame, *, output_dir: str | Path | None = None
) -> plt.Figure:
    """Show quality over node count, faceted by graph family."""
    frame = _completed(prepare_runs(runs)).dropna(subset=["n_nodes", "cut_fraction"])
    families = sorted(frame["graph_family"].unique()) if not frame.empty else []
    if not families:
        families = ["nessun dato"]
    columns = min(3, len(families))
    rows = int(np.ceil(len(families) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    for ax, family in zip(axes.flat, families):
        subset = frame.loc[frame["graph_family"].eq(family)]
        if subset.empty:
            _empty_axis(ax, "Nessun run completato")
            continue
        for (algorithm, backend), group in subset.groupby(["algorithm", "backend"]):
            grouped = group.groupby("n_nodes")["cut_fraction"]
            median = grouped.median()
            q1, q3 = grouped.quantile(0.25), grouped.quantile(0.75)
            ax.plot(median.index, median.values, marker="o", label=f"{algorithm} · {backend}")
            ax.fill_between(median.index, q1.values, q3.values, alpha=0.18)
        ax.set_title(str(family))
        ax.set_xlabel("Numero di nodi")
        ax.set_ylabel("Qualità: frazione del peso assoluto tagliato")
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
    for ax in axes.flat[len(families):]:
        ax.set_visible(False)
    fig.suptitle("Qualità della soluzione vs dimensione del grafo")
    fig.tight_layout()
    _save(fig, output_dir, "qualita_vs_dimensione_grafo")
    return fig


def plot_scaling(
    runs: pd.DataFrame, *, output_dir: str | Path | None = None
) -> tuple[plt.Figure, plt.Figure]:
    """Show runtime and measured energy scaling with node count."""
    frame = _completed(prepare_runs(runs)).dropna(subset=["n_nodes", "runtime_seconds"])
    fig_time, ax_time = plt.subplots(figsize=(9, 6))
    fig_energy, ax_energy = plt.subplots(figsize=(9, 6))
    if frame.empty:
        _empty_axis(ax_time, "Nessun run completato con tempo disponibile")
    else:
        for (algorithm, backend, family), group in frame.groupby(
            ["algorithm", "backend", "graph_family"], dropna=False
        ):
            grouped = group.groupby("n_nodes")["runtime_seconds"]
            median = grouped.median()
            ax_time.plot(median.index, median.values, marker="o", label=f"{algorithm} · {backend} · {family}")
        ax_time.set_xlabel("Numero di nodi")
        ax_time.set_ylabel("Tempo mediano per run (s)")
        ax_time.set_title("Scalabilità del tempo")
        ax_time.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax_time.grid(alpha=0.25)
    measured = frame.dropna(subset=["measured_energy_joules"])
    if measured.empty:
        _empty_axis(ax_energy, "Energia fisica non misurata: nessun dato da mostrare")
    else:
        for (algorithm, backend, family), group in measured.groupby(
            ["algorithm", "backend", "graph_family"], dropna=False
        ):
            median = group.groupby("n_nodes")["measured_energy_joules"].median()
            ax_energy.plot(median.index, median.values, marker="o", label=f"{algorithm} · {backend} · {family}")
        ax_energy.set_xlabel("Numero di nodi")
        ax_energy.set_ylabel("Energia fisica mediana misurata (J)")
        ax_energy.set_title("Scalabilità dell'energia misurata")
        ax_energy.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax_energy.grid(alpha=0.25)
    fig_time.tight_layout()
    fig_energy.tight_layout()
    _save(fig_time, output_dir, "tempo_vs_dimensione_grafo")
    _save(fig_energy, output_dir, "energia_vs_dimensione_grafo")
    return fig_time, fig_energy


def normalized_comparison(
    runs: pd.DataFrame, *, baseline: str = "exact"
) -> pd.DataFrame:
    """Add per-graph quality ratios and speedup against a CPU baseline.

    Quality is normalized against the exact CPU result when available, else
    against the best observed completed run. Speedup is algorithm/backend time
    relative to the median CPU time for the same graph and algorithm.
    """
    frame = _completed(prepare_runs(runs))
    if frame.empty:
        frame["quality_ratio"] = pd.Series(dtype=float)
        frame["speedup_vs_cpu"] = pd.Series(dtype=float)
        frame["quality_reference"] = pd.Series(dtype=str)
        return frame

    exact = frame.loc[
        frame["algorithm"].eq(baseline) & frame["backend"].eq("cpu")
    ].groupby("graph_id")["cut_value"].max()
    best = frame.groupby("graph_id")["cut_value"].max()
    frame["quality_reference_value"] = frame["graph_id"].map(exact)
    frame["quality_reference"] = np.where(
        frame["quality_reference_value"].notna(), f"{baseline} CPU",
        "miglior risultato osservato",
    )
    frame["quality_reference_value"] = frame["quality_reference_value"].fillna(
        frame["graph_id"].map(best)
    )
    denom = frame["quality_reference_value"]
    frame["quality_ratio"] = frame["cut_value"].div(denom.where(denom.abs() > 0))
    cpu_time = frame.loc[frame["backend"].eq("cpu")].groupby(
        ["graph_id", "algorithm"]
    )["runtime_seconds"].median()
    keys = pd.MultiIndex.from_frame(frame[["graph_id", "algorithm"]])
    frame["cpu_baseline_seconds"] = cpu_time.reindex(keys).to_numpy()
    frame["speedup_vs_cpu"] = frame["cpu_baseline_seconds"].div(
        frame["runtime_seconds"].where(frame["runtime_seconds"] > 0)
    )
    frame.loc[frame["backend"].eq("cpu"), "speedup_vs_cpu"] = 1.0
    return frame


def plot_normalized_comparison(
    runs: pd.DataFrame,
    *,
    baseline: str = "exact",
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot quality ratio and CPU-relative speedup by graph/algorithm/backend."""
    frame = normalized_comparison(runs, baseline=baseline)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if frame.empty:
        for ax in axes:
            _empty_axis(ax, "Nessun run completato")
    else:
        labels = frame["graph_id"].astype(str) + " · " + frame["algorithm"].astype(str) + " · " + frame["backend"].astype(str)
        axes[0].scatter(labels, frame["quality_ratio"], alpha=0.7)
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Cut / baseline")
        axes[0].set_title(f"Qualità normalizzata (Exact CPU, fallback best observed)")
        axes[1].scatter(labels, frame["speedup_vs_cpu"], alpha=0.7)
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Tempo CPU / tempo run")
        axes[1].set_title("Speedup rispetto alla CPU per stesso grafo e algoritmo")
        for ax in axes:
            ax.tick_params(axis="x", rotation=70)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir, "confronto_normalizzato")
    return frame, fig


def pareto_frontier(
    runs: pd.DataFrame,
    *,
    group_columns: Iterable[str] = ("graph_id",),
) -> pd.DataFrame:
    """Return completed, measured rows not dominated in time/energy/quality.

    Lower runtime and joules are better; higher cut fraction is better.
    Missing energy measurements are excluded instead of imputed.
    """
    frame = _completed(prepare_runs(runs)).dropna(
        subset=["runtime_seconds", "measured_energy_joules", "cut_fraction"]
    ).copy()
    groups = list(group_columns)
    if frame.empty:
        frame["pareto_optimal"] = pd.Series(dtype=bool)
        return frame
    if not groups:
        frame["_all"] = "all"
        groups = ["_all"]
    flags = pd.Series(False, index=frame.index)
    for _, group in frame.groupby(groups, dropna=False):
        values = group[["runtime_seconds", "measured_energy_joules", "cut_fraction"]].to_numpy(float)
        for local_index, point in zip(group.index, values):
            dominates = (
                (values[:, 0] <= point[0])
                & (values[:, 1] <= point[1])
                & (values[:, 2] >= point[2])
                & (
                    (values[:, 0] < point[0])
                    | (values[:, 1] < point[1])
                    | (values[:, 2] > point[2])
                )
            )
            flags.loc[local_index] = not bool(np.any(dominates))
    frame["pareto_optimal"] = flags
    return frame


def plot_pareto_frontier(
    runs: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot measured energy/runtime with quality encoded as color."""
    frame = pareto_frontier(runs)
    fig, ax = plt.subplots(figsize=(10, 6))
    if frame.empty:
        _empty_axis(ax, "Frontiera Pareto non disponibile senza misure fisiche valide")
    else:
        for graph_id, group in frame.groupby("graph_id", dropna=False):
            other = group.loc[~group["pareto_optimal"]]
            pareto = group.loc[group["pareto_optimal"]]
            ax.scatter(
                other["runtime_seconds"], other["measured_energy_joules"],
                c=other["cut_fraction"], cmap="viridis", alpha=0.45, marker="x",
            )
            scatter = ax.scatter(
                pareto["runtime_seconds"], pareto["measured_energy_joules"],
                c=pareto["cut_fraction"], cmap="viridis", edgecolors="black",
                s=80, label=str(graph_id),
            )
        fig.colorbar(scatter, ax=ax, label="Qualità: frazione del peso assoluto tagliato")
        ax.set_xlabel("Tempo per run (s), meno è meglio")
        ax.set_ylabel("Energia fisica misurata (J), meno è meglio")
        ax.set_title("Frontiera di Pareto: qualità come terzo obiettivo")
        ax.legend(title="Grafo")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir, "frontiera_pareto_tempo_energia_qualita")
    return frame, fig
