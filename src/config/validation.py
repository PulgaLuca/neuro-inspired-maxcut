from typing import List

from config.loader import SUPPORTED_ALGORITHMS, SUPPORTED_GRAPHS, SUPPORTED_HARDWARE


def validate_config(config: dict) -> List[str]:
    errors: List[str] = []

    errors.extend(_validate_experiment_configuration(config.get("experiment", {})))
    errors.extend(_validate_execution_configuration(config.get("execution", {})))
    errors.extend(_validate_graphs_configuration(config.get("graphs", [])))
    errors.extend(_validate_algorithms_configuration(config.get("algorithms", [])))
    errors.extend(_validate_backends_configuration(config.get("backends", [])))

    # Limitazioni dell'implementazione esatta: segnala in validazione, prima di lanciare.
    for graph in config.get("graphs", []):
        if isinstance(graph, dict) and isinstance(graph.get("sizes"), list):
            for algo in config.get("algorithms", []):
                if isinstance(algo, dict) and algo.get("name") == "exact":
                    limit = int(algo.get("params", {}).get("max_nodes", 24))
                    if any(isinstance(n, int) and n > limit for n in graph["sizes"]):
                        errors.append(f"Configured Exact with max_nodes={limit}, but '{graph.get('name')}' contains a superior size.")
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


def _validate_graphs_configuration(graphs: list) -> List[str]:
    errors: List[str] = []
    
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
        
        sizes = graph.get("sizes")
        if not isinstance(sizes, list) or not sizes or any(not isinstance(n, int) or n < 1 for n in sizes):
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
        
        if not isinstance(backend.get("params", {}), dict):
            errors.append(f"backends[{i}].params must be a map.")

    return errors
