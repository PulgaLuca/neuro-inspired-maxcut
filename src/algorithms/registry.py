"""Small registry of implemented algorithms and their backend capabilities."""
from collections.abc import Callable
from typing import Any

from algorithms.boltzmann import BoltzmannMachine
from algorithms.exact import ExactMaxCut
from algorithms.hopfield import HopfieldNetwork
from algorithms.ibm_qaoa import IBMQAOAMaxCut
from algorithms.local_search import LocalSearch
from algorithms.qaoa import QAOAMaxCut
from algorithms.spiking_neural_network import SpikingNeuralNetwork
from algorithms.simulated_annealing import SimulatedAnnealing

AlgorithmFactory = Callable[[dict[str, Any]], Any]

_ALGORITHMS: dict[str, AlgorithmFactory] = {
    "boltzmann": lambda params: BoltzmannMachine(**params),
    "exact": lambda params: ExactMaxCut(**params),
    "hopfield": lambda params: HopfieldNetwork(**params),
    "local_search": lambda params: LocalSearch(**params),
    "qaoa": lambda params: QAOAMaxCut(**params),
    "simulated_annealing": lambda params: SimulatedAnnealing(**params),
    "spiking_neural_network": lambda params: SpikingNeuralNetwork(**params),
}
_CAPABILITIES: dict[str, frozenset[str]] = {
    "boltzmann": frozenset({"cpu"}),
    "exact": frozenset({"cpu", "gpu"}),
    "hopfield": frozenset({"cpu"}),
    "local_search": frozenset({"cpu", "gpu"}),
    "qaoa": frozenset({"quantum_simulator"}),
    "simulated_annealing": frozenset({"cpu", "gpu"}),
    "spiking_neural_network": frozenset({"cpu"}),
}
_BACKEND_FACTORIES: dict[str, dict[str, AlgorithmFactory]] = {
    name: {backend: factory for backend in backends}
    for name, backends in _CAPABILITIES.items()
    for factory in (_ALGORITHMS[name],)
}
def register_algorithm(name: str, factory: AlgorithmFactory, *, backends: set[str]) -> None:
    """Register an implementation and the backends it really supports."""
    if not name or name in _ALGORITHMS:
        raise ValueError(f"Algorithm name is empty or already registered: {name!r}.")
    if not backends:
        raise ValueError("An algorithm must declare at least one supported backend.")
    _ALGORITHMS[name] = factory
    _CAPABILITIES[name] = frozenset(backends)
    _BACKEND_FACTORIES[name] = {backend: factory for backend in backends}


def register_backend_implementation(
    algorithm_name: str, backend_name: str, factory: AlgorithmFactory
) -> None:
    """Register a backend-specific adapter for an existing algorithm."""
    if algorithm_name not in _ALGORITHMS:
        raise ValueError(f"Algorithm is not registered: {algorithm_name!r}.")
    if not backend_name:
        raise ValueError("Backend name cannot be empty.")
    _BACKEND_FACTORIES.setdefault(algorithm_name, {})[backend_name] = factory
    _CAPABILITIES[algorithm_name] = _CAPABILITIES[algorithm_name] | {backend_name}


register_backend_implementation("qaoa", "quantum", lambda params: IBMQAOAMaxCut(**params))


def create_algorithm(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    hardware: str | None = None,
) -> Any:
    try:
        default_factory = _ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Algorithm is not implemented: {name!r}.") from exc
    if hardware is None:
        factory = default_factory
    else:
        try:
            factory = _BACKEND_FACTORIES[name][hardware]
        except KeyError as exc:
            raise ValueError(
                f"Algorithm {name!r} has no implementation for backend {hardware!r}."
            ) from exc
    return factory(dict(params or {}))


def supported_algorithms() -> frozenset[str]:
    return frozenset(_ALGORITHMS)


def supported_backends(name: str) -> frozenset[str]:
    return _CAPABILITIES.get(name, frozenset())
