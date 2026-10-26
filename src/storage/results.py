import csv, json, platform, sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graphs.base import to_networkx

def create_experiment_dir(config: dict, config_path: Path) -> Path:
    output = config["experiment"]["output"]
    root = Path(output["root"])
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    name = str(config["experiment"].get("name", "experiment"))
    target = root / f"{stamp}_{name}"
    suffix = 1
    while target.exists():
        target = root / f"{stamp}_{name}_{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    with (target / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for distribution in ("cupy-cuda12x", "cupy-cuda13x", "cupy-cuda11x", "cupy"):
        try:
            metadata["cupy"] = version(distribution)
            break
        except PackageNotFoundError:
            continue
    else:
        metadata["cupy"] = None
    if metadata["cupy"] is not None:
        try:
            import cupy

            metadata["cuda_runtime"] = int(cupy.cuda.runtime.runtimeGetVersion())
        except Exception as exc:
            metadata["cuda_runtime"] = None
            metadata["cuda_runtime_error"] = f"{type(exc).__name__}: {exc}"
    (target / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (target / "results.jsonl").touch()
    (target / "summary.csv").touch()
    return target

def save_run(run_dir: Path, problem, result, params: dict, graph_meta: dict,
             save_graph_image: bool = True, save_solution: bool = True,
             save_logs: bool = True) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"graph": graph_meta, "run": params}, f, sort_keys=False, allow_unicode=True)
    graph_data = {
        "name": graph_meta["graph_id"], "n_nodes": problem.n_variables,
        "edges": [
            {"u": u, "v": v, "weight": weight}
            for u, v, weight in problem.iter_edges()
        ],
    }
    if graph_meta.get("type") is not None:
        graph_data["type"] = graph_meta["type"]
    if graph_meta.get("instance") is not None:
        graph_data["instance"] = graph_meta["instance"]
    (run_dir / "graph.json").write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
    if save_graph_image:
        graph = to_networkx(problem)
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = __import__("networkx").spring_layout(graph, seed=graph_meta.get("graph_seed", 0))
        __import__("networkx").draw_networkx(graph, pos=pos, ax=ax, node_size=250, font_size=7, with_labels=True)
        ax.set_title(graph_meta["graph_id"])
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(run_dir / "graph.png", dpi=160)
        plt.close(fig)
    if save_solution:
        (run_dir / "solution.json").write_text(json.dumps({
            "solution_spins": np.asarray(result.solution).astype(int).tolist(),
            "objective_value": float(result.objective_value),
        }, indent=2), encoding="utf-8")
    result_data = result.to_dict()
    result_data.update({"graph_id": graph_meta["graph_id"], "n_nodes": problem.n_variables,
                        "n_edges": problem.n_edges, "density": graph_meta.get("density"),
                        "repetition": params["repetition"], "run_id": params["run_id"]})
    for key in ("type", "instance", "source_file"):
        if graph_meta.get(key) is not None:
            result_data["graph_type" if key == "type" else key] = graph_meta[key]
    (run_dir / "result.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    if save_logs:
        (run_dir / "run.log").write_text(
            f"status={result.status}\nalgorithm={result.algorithm}\nbackend={result.backend}\n"
            f"objective_value={result.objective_value}\nruntime_seconds={result.runtime_seconds}\n",
            encoding="utf-8")
    return result_data

def append_result(experiment_dir: Path, result_data: dict):
    with (experiment_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
    csv_path = experiment_dir / "summary.csv"
    flat = {k: v for k, v in result_data.items() if not isinstance(v, (dict, list))}
    physical_energy = result_data.get("physical_energy")
    if isinstance(physical_energy, dict):
        flat.update({
            f"physical_energy_{key}": value
            for key, value in physical_energy.items()
            if not isinstance(value, (dict, list))
        })
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    fieldnames = existing_fields + [
        key for key in flat if key not in existing_fields
    ]
    if fieldnames != existing_fields:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(flat)
