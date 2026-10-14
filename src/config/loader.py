from pathlib import Path
from typing import Any
import yaml

SUPPORTED_CONFIGURATIONS_PATH = (
    Path(__file__).resolve().parents[2]
    /"configs"
    /"supported_configurations.yaml"
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file (.yaml) not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Yaml file's root must be a map/object.")

    return config


def _load_supported_values(config: dict[str, Any], key: str) -> set[str]:
    values = config.get(key)

    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise ValueError(f"'{key}' in {SUPPORTED_CONFIGURATIONS_PATH} must be a non-empty list of strings.")
    
    return set(values)


def load_supported_configurations() -> dict[str, set[str]]:
    """Load and validate supported graph, algorithm, and hardware names."""
    config = load_config(SUPPORTED_CONFIGURATIONS_PATH)
    keys = ("supported_graphs", "supported_algorithms", "supported_hardware")
    return {key: _load_supported_values(config, key) for key in keys}


_SUPPORTED_CONFIGURATIONS = load_supported_configurations()
SUPPORTED_GRAPHS = _SUPPORTED_CONFIGURATIONS["supported_graphs"]
SUPPORTED_ALGORITHMS = _SUPPORTED_CONFIGURATIONS["supported_algorithms"]
SUPPORTED_HARDWARE = _SUPPORTED_CONFIGURATIONS["supported_hardware"]
