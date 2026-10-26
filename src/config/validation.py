from typing import List
from math import isfinite

from config.loader import (
    SUPPORTED_GRAPHS,
    SUPPORTED_HARDWARE,
    load_gset,
    resolve_gset_instances,
)
from algorithms.registry import supported_algorithms


def validate_config(config: dict) -> List[str]:
    errors: List[str] = []
    graph_sizes: dict[int, list[int]] = {}

    errors.extend(_validate_experiment_configuration(config.get("experiment", {})))
    errors.extend(_validate_execution_configuration(config.get("execution", {})))
    for section in ("graphs", "algorithms", "backends"):
        value = config.get(section)
        if not isinstance(value, list) or not value:
            errors.append(f"'{section}' must be a non-empty list.")
    errors.extend(_validate_graphs_configuration(config.get("graphs", []), graph_sizes))
    errors.extend(_validate_algorithms_configuration(config.get("algorithms", [])))
    errors.extend(_validate_backends_configuration(config.get("backends", [])))

    backend_names = {
        backend.get("name") for backend in config.get("backends", [])
        if isinstance(backend, dict)
    }
    for index, algorithm in enumerate(config.get("algorithms", [])):
        if isinstance(algorithm, dict) and algorithm.get("name") == "qaoa":
            if not backend_names.intersection({"quantum_simulator", "quantum"}):
                errors.append(
                    f"algorithms[{index}] requires 'quantum_simulator' or a configured "
                    "'quantum' hardware backend."
                )

    for graph_index, graph in enumerate(config.get("graphs", [])):
        if isinstance(graph, dict):
            for algo in config.get("algorithms", []):
                if isinstance(algo, dict) and algo.get("name") == "exact":
                    exact_params = algo.get("params", {})
                    if not isinstance(exact_params, dict):
                        exact_params = {}
                    configured_limit = exact_params.get("max_nodes", 24)
                    limit = configured_limit if isinstance(configured_limit, int) and not isinstance(configured_limit, bool) else 24
                    sizes = graph_sizes.get(graph_index, graph.get("sizes", []))
                    if isinstance(sizes, list) and any(
                        isinstance(n, int) and n > limit for n in sizes
                    ):
                        errors.append(
                            f"Configured Exact with max_nodes={limit}, but "
                            f"'{graph.get('name')}' contains a superior size."
                        )
            for algo_index, algo in enumerate(config.get("algorithms", [])):
                if not isinstance(algo, dict) or algo.get("name") != "qaoa":
                    continue
                if "quantum_simulator" not in backend_names:
                    continue
                qaoa_params = algo.get("params", {})
                if not isinstance(qaoa_params, dict):
                    qaoa_params = {}
                configured_limit = qaoa_params.get("max_nodes", 20)
                limit = (
                    configured_limit
                    if isinstance(configured_limit, int)
                    and not isinstance(configured_limit, bool)
                    else 20
                )
                sizes = graph_sizes.get(graph_index, graph.get("sizes", []))
                if isinstance(sizes, list) and any(
                    isinstance(size, int) and not isinstance(size, bool)
                    and size > limit
                    for size in sizes
                ):
                    errors.append(
                        f"algorithms[{algo_index}] state-vector simulation is "
                        f"limited to max_nodes={limit}, but graph "
                        f"'{graph.get('name')}' exceeds that limit."
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
        
        if name not in supported_algorithms():
            errors.append(f"{prefix}.name not implemented: {name!r}.")
        
        if not isinstance(algo.get("params", {}), dict):
            errors.append(f"{prefix}.params must be a map.")
        
        params = algo.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if name == "exact":
            max_nodes = params.get("max_nodes", 24)
            if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
                errors.append(f"{prefix}.params.max_nodes must be an integer >= 1.")
            batch_size = params.get("gpu_batch_size", 4096)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                errors.append(f"{prefix}.params.gpu_batch_size must be an integer >= 1.")
        
        if name == "simulated_annealing":
            steps = params.get("steps", 10000)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                errors.append(f"{prefix}.params.steps must be an integer >= 1.")

            try:
                ti = float(params.get("initial_temperature", 10))
                tf = float(params.get("final_temperature", 0.01))
            except (TypeError, ValueError):
                ti, tf = 0.0, 0.0
            if ti <= 0 or tf <= 0 or tf > ti:
                errors.append(f"{prefix}: invalid temperature (0 < final_temperature <= initial_temperature).")
            replicas = params.get("gpu_replicas", 128)
            if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
                errors.append(f"{prefix}.params.gpu_replicas must be an integer >= 1.")

        if name == "local_search" and isinstance(params, dict):
            max_iterations = params.get("max_iterations", 10000)
            if (
                isinstance(max_iterations, bool)
                or not isinstance(max_iterations, int)
                or max_iterations < 1
            ):
                errors.append(
                    f"{prefix}.params.max_iterations must be an integer >= 1."
                )

        if name == "hopfield":
            max_iterations = params.get("max_iterations", 100_000)
            if (
                isinstance(max_iterations, bool)
                or not isinstance(max_iterations, int)
                or max_iterations < 1
            ):
                errors.append(
                    f"{prefix}.params.max_iterations must be an integer >= 1."
                )

        if name == "boltzmann":
            steps = params.get("steps", 50_000)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                errors.append(f"{prefix}.params.steps must be an integer >= 1.")
            temperature = params.get("temperature", 1.0)
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not isfinite(temperature)
                or temperature <= 0
            ):
                errors.append(f"{prefix}.params.temperature must be finite and > 0.")

        if name == "spiking_neural_network":
            time_steps = params.get("time_steps", 1000)
            if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps < 1:
                errors.append(f"{prefix}.params.time_steps must be an integer >= 1.")
            for key, default in (("threshold", 1.0), ("synaptic_scale", 0.1)):
                value = params.get(key, default)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value <= 0
                ):
                    errors.append(f"{prefix}.params.{key} must be finite and > 0.")
            decay = params.get("membrane_decay", 0.2)
            if (
                isinstance(decay, bool)
                or not isinstance(decay, (int, float))
                or not isfinite(decay)
                or not 0 <= decay < 1
            ):
                errors.append(
                    f"{prefix}.params.membrane_decay must be finite and in [0, 1)."
                )
            refractory_steps = params.get("refractory_steps", 1)
            if (
                isinstance(refractory_steps, bool)
                or not isinstance(refractory_steps, int)
                or refractory_steps < 0
            ):
                errors.append(
                    f"{prefix}.params.refractory_steps must be an integer >= 0."
                )

        if name == "qaoa":
            for key, default in (
                ("depth", 1),
                ("optimizer_iterations", 80),
                ("max_nodes", 20),
                ("optimizer_restarts", 1),
                ("shots", 1024),
            ):
                value = params.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    errors.append(f"{prefix}.params.{key} must be an integer >= 1.")

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
        if backend.get("name") == "quantum":
            for key in ("backend_name", "channel", "instance", "token_env_var"):
                value = params.get(key)
                if value is not None and not isinstance(value, str):
                    errors.append(f"{prefix}.params.{key} must be a string when set.")
            if params.get("channel", "ibm_quantum_platform") not in {
                "ibm_quantum_platform", "ibm_cloud", "ibm_quantum"
            }:
                errors.append(
                    f"{prefix}.params.channel must be ibm_quantum_platform, "
                    "ibm_cloud, or ibm_quantum."
                )

    return errors
