import csv, json, platform, sys
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
            {"u": int(i), "v": int(j), "weight": float(problem.weights[i, j])}
            for i in range(problem.n_variables) for j in range(i + 1, problem.n_variables)
            if problem.weights[i, j] != 0
        ],
    }
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
                        "n_edges": len(graph_data["edges"]), "density": graph_meta.get("density"),
                        "repetition": params["repetition"], "run_id": params["run_id"]})
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
    exists = csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()), extrasaction="ignore")
        if not exists: writer.writeheader()
        writer.writerow(flat)
