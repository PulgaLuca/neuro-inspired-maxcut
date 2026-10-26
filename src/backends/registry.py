"""Registry of configured hardware targets and locally available providers."""
from dataclasses import dataclass

from backends.cupy_backend import gpu_available
from backends.ibm_quantum import quantum_available


@dataclass(frozen=True)
class BackendCapability:
    name: str
    available: bool
    reason: str | None = None


_BACKENDS: dict[str, BackendCapability] = {
    "cpu": BackendCapability("cpu", True),
    "quantum_simulator": BackendCapability("quantum_simulator", True),
    "neuromorphic": BackendCapability(
        "neuromorphic", False, "No neuromorphic provider is configured."
    ),
    "quantum": BackendCapability("quantum", False, "No quantum provider is configured."),
}


def register_backend(capability: BackendCapability) -> None:
    """Register a hardware target and its real provider availability."""
    if not capability.name:
        raise ValueError("Backend name cannot be empty.")
    existing = _BACKENDS.get(capability.name)
    if existing is not None and existing.available:
        raise ValueError(f"Backend is already registered: {capability.name!r}.")
    _BACKENDS[capability.name] = capability


def get_backend(name: str) -> BackendCapability | None:
    if name == "gpu":
        available, reason = gpu_available()
        return BackendCapability("gpu", available, reason)
    if name == "quantum":
        available, reason = quantum_available()
        return BackendCapability("quantum", available, reason)
    return _BACKENDS.get(name)
