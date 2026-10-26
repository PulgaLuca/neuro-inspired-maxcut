"""Optional IBM Quantum Runtime integration with local-only credentials."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


DEFAULT_TOKEN_FILE = Path(__file__).resolve().parents[1] / "apikey.json"


def resolve_ibm_token(
    *, env: dict[str, str] | None = None, token_file: str | Path | None = None
) -> str | None:
    """Read the IBM token from an environment variable or a local ignored file.

    The token value is returned only to the caller and must never be logged,
    serialized, or copied into experiment metadata.
    """
    environment = os.environ if env is None else env
    token = environment.get("IBM_QUANTUM_TOKEN", "").strip()
    if token:
        return token

    configured_path = environment.get("IBM_QUANTUM_TOKEN_FILE")
    path = Path(token_file or configured_path or DEFAULT_TOKEN_FILE)
    if not path.is_file():
        return None
    try:
        credential = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = credential.get("apikey") if isinstance(credential, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def quantum_available() -> tuple[bool, str | None]:
    """Report local SDK and credential readiness without connecting remotely."""
    try:
        sdk_available = all(
            importlib.util.find_spec(module) is not None
            for module in ("qiskit", "qiskit_ibm_runtime")
        )
    except (ImportError, ValueError):
        sdk_available = False
    if not sdk_available:
        return False, "IBM Qiskit SDK missing; install the optional 'quantum' extra."
    if resolve_ibm_token() is None:
        return False, (
            "IBM Quantum token missing; set IBM_QUANTUM_TOKEN or configure "
            "IBM_QUANTUM_TOKEN_FILE."
        )
    return True, None
