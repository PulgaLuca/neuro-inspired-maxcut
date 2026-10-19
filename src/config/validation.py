from typing import List

from config.loader import (
    SUPPORTED_ALGORITHMS,
    SUPPORTED_GRAPHS,
    SUPPORTED_HARDWARE,
    load_gset,
    resolve_gset_instances,
)


def validate_config(config: dict) -> List[str]:
    errors: List[str] = []
    graph_sizes: dict[int, list[int]] = {}

    errors.extend(_validate_experiment_configuration(config.get("experiment", {})))
    errors.extend(_validate_execution_configuration(config.get("execution", {})))
    errors.extend(_validate_graphs_configuration(config.get("graphs", []), graph_sizes))
    errors.extend(_validate_algorithms_configuration(config.get("algorithms", [])))
    errors.extend(_validate_backends_configuration(config.get("backends", [])))

    for graph_index, graph in enumerate(config.get("graphs", [])):
        if isinstance(graph, dict):
            for algo in config.get("algorithms", []):
                if isinstance(algo, dict) and algo.get("name") == "exact":
                    limit = int(algo.get("params", {}).get("max_nodes", 24))
                    sizes = graph_sizes.get(graph_index, graph.get("sizes", []))
                    if isinstance(sizes, list) and any(
                        isinstance(n, int) and n > limit for n in sizes
                    ):
                        errors.append(
                            f"Configured Exact with max_nodes={limit}, but "
                            f"'{graph.get('name')}' contains a superior size."
                        )
    return errors


def _validate_experiment_configuration(experiment: dict) -> List[str]:
    errors: List[str] = []

    if not isinstance(experiment, dict):
        errors.append("'experiment' must be a map.")
        experiment = {}

    if not isinstance(experiment.get("seed", 99), int):
        errors.append("'experiment.seed' must be an integer.")
    
    if not isinstance(experiment.get("repetitions", 1), int) or experiment.get("repetitions", 1) < 1:
        errors.append("'experiment.repetitions' must be an integer >= 1.")
    
    output = experiment.get("output", {})
    if not isinstance(output, dict) or not isinstance(output.get("root", "results"), str):
        errors.append("'experiment.output.root' must be a string.")

    return errors


def _validate_execution_configuration(execution: dict) -> List[str]:
    errors: List[str] = []
    
    if not isinstance(execution, dict) or float(execution.get("time_limit_seconds", 60)) <= 0:
        errors.append("'execution.time_limit_seconds' must be greater than 0.")

    return errors


def _validate_graphs_configuration(
    graphs: list, graph_sizes: dict[int, list[int]] | None = None
) -> List[str]:
    errors: List[str] = []

    if not isinstance(graphs, list):
        return ["'graphs' must be a list."]
    
    for i, graph in enumerate(graphs):
        prefix = f"graphs[{i}]"
        if not isinstance(graph, dict):
            errors.append(f"{prefix} must be a map.")
            continue
        
        name, kind = graph.get("name"), graph.get("type")
        if not name: 
            errors.append(f"{prefix}.name is mandatory.")
        
        if kind not in SUPPORTED_GRAPHS:
            errors.append(f"{prefix}.type not supported: {kind!r}.")
        
        if kind == "gset":
            instances = graph.get("instances", "all")
            directory = graph.get("path")
            if directory is not None and not isinstance(directory, str):
                errors.append(f"{prefix}.path must be a string.")
                continue
            try:
                selected = resolve_gset_instances(instances, directory)
                sizes = []
                for instance in selected:
                    sizes.append(load_gset(instance, directory)[instance].n_variables)
                if graph_sizes is not None:
                    graph_sizes[i] = sizes
            except (FileNotFoundError, OSError, ValueError) as exc:
                errors.append(f"{prefix}.instances: {exc}")
        else:
            sizes = graph.get("sizes")
            if not isinstance(sizes, list) or not sizes or any(
                not isinstance(n, int) or n < 1 for n in sizes
            ):
                errors.append(f"{prefix}.sizes must be a positive integer list.")
        
        density = graph.get("density", 0.3)
        if kind == "erdos_renyi" and (not isinstance(density, (int, float)) or not 0 <= density <= 1):
            errors.append(f"{prefix}.density must be between 0 and 1.")
        
        if not isinstance(graph.get("weighted", False), bool):
            errors.append(f"{prefix}.weighted must be a boolean.")

    return errors


def _validate_algorithms_configuration(algorithms: list) -> List[str]:
    errors: List[str] = []

    for i, algo in enumerate(algorithms):
        prefix = f"algorithms[{i}]"

        if not isinstance(algo, dict): 
            errors.append(f"{prefix} must be a map.")
            continue
        name = algo.get("name")
        
        if name not in SUPPORTED_ALGORITHMS:
            errors.append(f"{prefix}.name not implemented: {name!r}.")
        
        if not isinstance(algo.get("params", {}), dict):
            errors.append(f"{prefix}.params must be a map.")
        
        params = algo.get("params", {})
        if name == "exact":
            if int(params.get("max_nodes", 24)) < 1:
                errors.append(f"{prefix}.params.max_nodes must be >= 1.")
        
        if name == "simulated_annealing":
            if int(params.get("steps", 10000)) < 1:
                errors.append(f"{prefix}.params.steps must be >= 1.")
        
            ti, tf = float(params.get("initial_temperature", 10)), float(params.get("final_temperature", 0.01))
            if ti <= 0 or tf <= 0 or tf > ti:
                errors.append(f"{prefix}: invalid temperature (0 < final_temperature <= initial_temperature).")

    return errors


def _validate_backends_configuration(backends: list) -> List[str]:
    errors: List[str] = []

    for i, backend in enumerate(backends):
        if not isinstance(backend, dict): 
            errors.append(f"backends[{i}] must be a map.")
            continue
        
        if backend.get("name") not in SUPPORTED_HARDWARE:
            errors.append(f"backends[{i}].name not supported: {backend.get('name')!r}.")
        
        params = backend.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"backends[{i}].params must be a map.")
            continue

        device_index = params.get("device_index", 0)
        if backend.get("name") == "gpu" and (
            isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
        ):
            errors.append(
                f"backends[{i}].params.device_index must be a non-negative integer."
            )

    return errors
