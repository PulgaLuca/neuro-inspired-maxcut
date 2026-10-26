"""Lazy CuPy helpers; importing the CPU framework never requires CUDA."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from scipy import sparse


def require_cupy(device_index: int = 0) -> tuple[Any, str]:
    """Return CuPy and the selected device name, or raise a diagnostic error."""
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("GPU device_index must be a non-negative integer.")
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU execution requires the optional CuPy dependency. Install with "
            "`pip install -e '.[gpu]'` in a CUDA 12 environment."
        ) from exc

    try:
        count = cp.cuda.runtime.getDeviceCount()
        if device_index >= count:
            raise RuntimeError(
                f"GPU device_index {device_index} is unavailable; CUDA detected {count} device(s)."
            )
        device = cp.cuda.Device(device_index)
        device.use()
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        return cp, str(device_name)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize CUDA GPU {device_index}: {exc}") from exc


@lru_cache(maxsize=1)
def gpu_available() -> tuple[bool, str | None]:
    try:
        require_cupy(0)
    except (ImportError, RuntimeError, ValueError) as exc:
        return False, str(exc)
    return True, None


def to_cupy_csr(graph, cp):
    """Copy a MaxCut adjacency matrix to a CuPy CSR matrix."""
    from cupyx.scipy import sparse as cupy_sparse

    matrix = sparse.csr_matrix(graph.weights, dtype="float64")
    return cupy_sparse.csr_matrix(
        (cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
        shape=matrix.shape,
    )
