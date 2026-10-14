import time
from pathlib import Path
import numpy as np
import json
from algorithms.exact import ExactMaxCut
from algorithms.simulated_annealing import SimulatedAnnealing
from graphs.erdos_renyi import erdos_renyi
from storage.results import save_run, append_result

def _build_algorithm(spec):
    name, params = spec["name"], dict(spec.get("params", {}))
    
    if name == "exact": 
        return ExactMaxCut(**params)
    
    if name == "simulated_annealing": 
        return SimulatedAnnealing(**params)
    
    raise ValueError(f"Algorithm not implemented yet: {name}")

def plan_runs(config: dict) -> list[dict]:
    plan = []
    rep_count = config["experiment"]["repetitions"]
    for graph_spec in config["graphs"]:
        for size in graph_spec["sizes"]:
            for algorithm_spec in config["algorithms"]:
                for backend_spec in config["backends"]:
                    for rep in range(rep_count):
                        plan.append({"graph": graph_spec, "size": size, "algorithm": algorithm_spec, "backend": backend_spec, "repetition": rep + 1})
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
        graph_seed = base_seed + sum((i + 1) * ord(c) for i, c in enumerate(str(gs["name"]))) + n
        graph_key = (gs["name"], n, graph_seed, gs.get("density", 0.3), gs.get("weighted", False))
        graph_id = f"{gs['name']}_n{n}"
        
        run_seed = base_seed + idx * 1009 + rep
        run_id = f"run_{idx:06d}"
        run_dir = experiment_dir/"runs"/run_id
        params = {
            "run_id": run_id, "repetition": rep, "seed": run_seed,
            "algorithm": alg_spec, "backend": backend_spec,
            "time_limit_seconds": time_limit,
        }
        graph_meta = {
            "graph_id": graph_id, "type": gs["type"], "size": n,
            "density": gs.get("density"), "weighted": gs.get("weighted", False),
            "graph_seed": graph_seed,
        }
        try:
            if graph_key not in graph_cache:
                if gs["type"] != "erdos_renyi":
                    raise ValueError(f"Tipo di grafo non implementato: {gs['type']}")
                
                graph_cache[graph_key] = erdos_renyi(
                    n=n, p=float(gs.get("density", 0.3)), seed=graph_seed,
                    weighted=bool(gs.get("weighted", False)))
            
            graph = graph_cache[graph_key]
            algorithm = _build_algorithm(alg_spec)
            
            start = time.perf_counter()
            result = algorithm.solve(
                graph,
                hardware=backend_spec["name"],
                seed=run_seed,
                time_limit_seconds=time_limit,
            )
            wall_time = time.perf_counter() - start
            
            # Verifica l'obiettivo contro l'implementazione canonica del problema.
            recomputed = graph.evaluate(result.solution)
            if not np.isclose(recomputed, result.objective_value, atol=1e-7):
                raise ValueError(f"Objective mismatch: result={result.objective_value}, recomputed={recomputed}")
            
            result.backend = backend_spec["name"]
            result.runtime_seconds = wall_time
            result_data = save_run(
                run_dir, graph, result, params, graph_meta,
                save_graph_image=bool(output.get("save_graph_image", True)),
                save_solution=bool(output.get("save_solution", True)),
                save_logs=bool(output.get("save_logs", True)))
            
            result_data.update({"algorithm": alg_spec["name"], "backend": backend_spec["name"],
                                "seed": run_seed, "graph_seed": graph_seed})

            # Aggiorna il result JSON con campi tabellari completi.
            (run_dir/"result.json").write_text(json.dumps(result_data, indent=2), encoding="utf-8")
            append_result(experiment_dir, result_data)
            counts[result.status if result.status in counts else "completed"] += 1

        except Exception as exc:
            counts["failed"] += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir/"error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            
            if config["execution"].get("fail_fast", False):
                raise
            
    return counts
