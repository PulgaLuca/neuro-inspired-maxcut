import time
from pathlib import Path
import numpy as np
import json
from algorithms.registry import create_algorithm, supported_backends
from backends.registry import get_backend
from graphs.erdos_renyi import erdos_renyi
from config.loader import load_gset, resolve_gset_instances
from graphs.gset import gset_directory
from metrics.physical_energy import PhysicalEnergyMeter
from storage.results import save_run, append_result


def _build_algorithm(spec, hardware: str):
    return create_algorithm(spec["name"], spec.get("params", {}), hardware=hardware)


def derive_run_seed(base_seed: int, graph_id: str, repetition: int) -> int:
    """Use one algorithm seed per graph/repetition across comparison dimensions."""
    graph_component = sum((index + 1) * ord(char) for index, char in enumerate(graph_id))
    return base_seed + graph_component + repetition * 1009


def plan_runs(config: dict) -> list[dict]:
    plan = []
    rep_count = config["experiment"]["repetitions"]
    for graph_spec in config["graphs"]:
        if graph_spec["type"] == "gset":
            graph_instances = (
                (None, instance)
                for instance in resolve_gset_instances(
                    graph_spec.get("instances", "all"), graph_spec.get("path")
                )
            )
        else:
            graph_instances = ((size, None) for size in graph_spec["sizes"])

        for size, instance in graph_instances:
            for algorithm_spec in config["algorithms"]:
                for backend_spec in config["backends"]:
                    for rep in range(rep_count):
                        backend = get_backend(backend_spec["name"])
                        algorithm_supports = backend_spec["name"] in supported_backends(
                            algorithm_spec["name"]
                        )
                        supported = bool(backend and backend.available and algorithm_supports)
                        reasons = []
                        if not backend:
                            reasons.append("no backend provider is registered")
                        elif not backend.available:
                            reasons.append(backend.reason or "backend provider unavailable")
                        if not algorithm_supports:
                            reasons.append(
                                f"{algorithm_spec['name']} has no {backend_spec['name']} implementation"
                            )
                        plan.append({
                            "graph": graph_spec,
                            "size": size,
                            "instance": instance,
                            "algorithm": algorithm_spec,
                            "backend": backend_spec,
                            "repetition": rep + 1,
                            "supported": supported,
                            "compatibility": "supported" if supported else "; ".join(reasons),
                        })
    return plan


def execute_plan(config: dict, config_path: Path, experiment_dir: Path, plan: list[dict]) -> dict:
    base_seed = int(config["experiment"]["seed"])
    time_limit = float(config["execution"]["time_limit_seconds"])
    output = config["experiment"]["output"]
    counts = {"completed": 0, "time_limit": 0, "failed": 0}
    graph_cache = {}

    for idx, item in enumerate(plan, start=1):
        gs, alg_spec, backend_spec = item["graph"], item["algorithm"], item["backend"]
        rep, n = item["repetition"], item["size"]
        instance = item.get("instance")
        if gs["type"] == "gset":
            graph_seed = None
            graph_id = instance
            graph_key = ("gset", str(gset_directory(gs.get("path"))), instance)
        else:
            graph_seed = (
                base_seed
                + sum((i + 1) * ord(c) for i, c in enumerate(str(gs["name"])))
                + n
            )
            graph_id = f"{gs['name']}_n{n}"
            graph_key = (
                gs["name"], n, graph_seed, gs.get("density", 0.3),
                gs.get("weighted", False),
            )
        
        run_seed = derive_run_seed(base_seed, graph_id, rep)
        run_id = f"run_{idx:06d}"
        run_dir = experiment_dir/"runs"/run_id
        params = {
            "run_id": run_id, "repetition": rep, "seed": run_seed,
            "algorithm": alg_spec, "backend": backend_spec,
            "time_limit_seconds": time_limit,
        }
        graph_meta = {
            "graph_id": graph_id,
            "type": gs["type"],
            "size": n,
            "density": gs.get("density"),
            "weighted": gs.get("weighted", False),
        }
        if graph_seed is not None:
            graph_meta["graph_seed"] = graph_seed
        if instance is not None:
            graph_meta["instance"] = instance
            graph_meta["source_file"] = str(
                gset_directory(gs.get("path")) / instance / f"{instance}.txt"
            )
        try:
            if not item.get("supported", True):
                raise RuntimeError(item.get("compatibility", "Unsupported algorithm/backend combination"))
            if graph_key not in graph_cache:
                if gs["type"] == "gset":
                    graph_cache[graph_key] = load_gset(
                        instance, directory=gs.get("path")
                    )[instance]
                    graph_meta["size"] = graph_cache[graph_key].n_variables
                elif gs["type"] == "erdos_renyi":
                    graph_cache[graph_key] = erdos_renyi(
                        n=n, p=float(gs.get("density", 0.3)), seed=graph_seed,
                        weighted=bool(gs.get("weighted", False)))
                else:
                    raise ValueError(f"Tipo di grafo non implementato: {gs['type']}")
            
            graph = graph_cache[graph_key]
            graph_meta["size"] = graph.n_variables
            algorithm = _build_algorithm(alg_spec, backend_spec["name"])
            
            energy_meter = PhysicalEnergyMeter.start(
                backend_spec["name"], backend_spec.get("params", {})
            )
            start = time.perf_counter()
            try:
                result = algorithm.solve(
                    graph,
                    hardware=backend_spec["name"],
                    seed=run_seed,
                    time_limit_seconds=time_limit,
                    **backend_spec.get("params", {}),
                )
                wall_time = time.perf_counter() - start
            finally:
                physical_energy = energy_meter.finish()
            
            # Verifica l'obiettivo contro l'implementazione canonica del problema.
            recomputed = graph.evaluate(result.solution)
            if not np.isclose(recomputed, result.objective_value, atol=1e-7):
                raise ValueError(f"Objective mismatch: result={result.objective_value}, recomputed={recomputed}")
            result.objective_value = float(recomputed)
            
            result.backend = backend_spec["name"]
            result.runtime_seconds = wall_time
            result_data = save_run(
                run_dir, graph, result, params, graph_meta,
                save_graph_image=bool(output.get("save_graph_image", True)),
                save_solution=bool(output.get("save_solution", True)),
                save_logs=bool(output.get("save_logs", True)))
            
            result_data.update({"algorithm": alg_spec["name"], "backend": backend_spec["name"],
                                "seed": run_seed, "physical_energy": physical_energy,
                                "schema_version": 1, "framework_version": "0.1.0",
                                "algorithm_params": alg_spec.get("params", {}),
                                "backend_params": backend_spec.get("params", {}),
                                "provider": result.metadata.get(
                                    "provider",
                                    {"cpu": "local_cpu", "gpu": "cupy_cuda",
                                     "quantum_simulator": "numpy_statevector_simulator"}.get(
                                        backend_spec["name"]
                                    ),
                                ),
                                "device": result.metadata.get(
                                    "device", "CPU" if backend_spec["name"] in
                                    {"cpu", "quantum_simulator"} else None
                                )})
            if graph_seed is not None:
                result_data["graph_seed"] = graph_seed

            # Aggiorna il result JSON con campi tabellari completi.
            (run_dir/"result.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
            append_result(experiment_dir, result_data)
            counts[result.status if result.status in counts else "completed"] += 1

        except Exception as exc:
            counts["failed"] += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir/"error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            failure = {
                "schema_version": 1, "framework_version": "0.1.0",
                "run_id": run_id, "graph_id": graph_id,
                "graph_type": gs["type"], "instance": instance,
                "n_nodes": graph_meta.get("size"), "algorithm": alg_spec["name"],
                "algorithm_params": alg_spec.get("params", {}),
                "backend": backend_spec["name"], "backend_params": backend_spec.get("params", {}),
                "provider": None, "device": None,
                "seed": run_seed, "graph_seed": graph_seed,
                "repetition": rep, "time_limit_seconds": time_limit,
                "source_file": graph_meta.get("source_file"),
                "status": "failed", "solution": None,
                "objective_value": None, "runtime_seconds": None,
                "physical_energy": {"joules": None, "unit": "J", "status": "unavailable",
                                    "method": "not_measured", "scope": "device",
                                    "error": "Run did not produce an execution result."},
                "error": f"{type(exc).__name__}: {exc}",
            }
            (run_dir/"result.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            append_result(experiment_dir, failure)
            
            if config["execution"].get("fail_fast", False):
                raise
            
    return counts
