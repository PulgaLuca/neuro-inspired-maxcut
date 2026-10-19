from pathlib import Path
import re
from typing import Sequence

import numpy as np
from scipy import sparse

from problems.maxcut import MaxCutProblem


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GSET_DIRECTORY = REPOSITORY_ROOT / "istances"
_INSTANCE_PATTERN = re.compile(r"G(\d+)\Z")


def gset_directory(directory: str | Path | None = None) -> Path:
    """Return the GSET directory, resolving relative paths from the repository root."""
    if directory is None:
        return GSET_DIRECTORY
    path = Path(directory)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def resolve_gset_instances(
    instances: str | Sequence[str],
    directory: str | Path | None = None,
) -> list[str]:
    """Resolve one, several, or all available GSET instance identifiers."""
    root = gset_directory(directory)
    available = {
        path.parent.name
        for path in root.glob("G*/*.txt")
        if path.stem == path.parent.name
        and _INSTANCE_PATTERN.fullmatch(path.parent.name)
    }
    ordered_available = sorted(available, key=lambda value: int(value[1:]))

    if isinstance(instances, str):
        selected = ordered_available if instances.lower() == "all" else [instances]
    elif isinstance(instances, Sequence) and not isinstance(instances, (bytes, bytearray)):
        selected = list(instances)
    else:
        raise ValueError("'instances' must be an instance name, a list of names, or 'all'.")

    if not selected:
        raise ValueError("At least one GSET instance must be selected.")
    if any(not isinstance(instance, str) or not _INSTANCE_PATTERN.fullmatch(instance)
           for instance in selected):
        raise ValueError("GSET instance names must use the format 'G<number>'.")
    if len(set(selected)) != len(selected):
        raise ValueError("GSET instance selection contains duplicates.")

    missing = [instance for instance in selected if instance not in available]
    if missing:
        raise FileNotFoundError(
            f"GSET instance file(s) not found in {root}: {', '.join(missing)}"
        )
    return selected


def _load_gset_instance(instance: str, directory: Path) -> MaxCutProblem:
    path = directory / instance / f"{instance}.txt"
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}:1: expected '<node_count> <edge_count>'.")
        try:
            node_count, edge_count = map(int, header)
        except ValueError as exc:
            raise ValueError(f"{path}:1: node and edge counts must be integers.") from exc
        if node_count < 1 or edge_count < 0:
            raise ValueError(f"{path}:1: node count must be positive and edge count non-negative.")

        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        seen_edges: set[tuple[int, int]] = set()
        for line_number, line in enumerate(stream, start=2):
            content = line.strip()
            if not content or content.startswith("#"):
                continue
            fields = content.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected '<u> <v> <weight>'.")
            try:
                u, v = int(fields[0]), int(fields[1])
                weight = float(fields[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid edge values.") from exc
            if not np.isfinite(weight):
                raise ValueError(f"{path}:{line_number}: edge weight must be finite.")
            if not (1 <= u <= node_count and 1 <= v <= node_count):
                raise ValueError(
                    f"{path}:{line_number}: node indices must be between 1 and {node_count}."
                )
            if u == v:
                raise ValueError(f"{path}:{line_number}: self-loops are not supported.")
            edge = (min(u, v), max(u, v))
            if edge in seen_edges:
                raise ValueError(f"{path}:{line_number}: duplicate edge {edge}.")
            seen_edges.add(edge)

            zero_based_u, zero_based_v = u - 1, v - 1
            rows.extend((zero_based_u, zero_based_v))
            columns.extend((zero_based_v, zero_based_u))
            weights.extend((weight, weight))

    if len(seen_edges) != edge_count:
        raise ValueError(
            f"{path}: header declares {edge_count} edges, but {len(seen_edges)} were read."
        )

    adjacency = sparse.coo_matrix(
        (weights, (rows, columns)), shape=(node_count, node_count), dtype=np.float64
    ).tocsr()
    return MaxCutProblem(weights=adjacency)


def load_gset(
    instances: str | Sequence[str] = "all",
    directory: str | Path | None = None,
) -> dict[str, MaxCutProblem]:
    """Load selected GSET instances as sparse weighted Max-Cut problems."""
    root = gset_directory(directory)
    selected = resolve_gset_instances(instances, root)
    return {instance: _load_gset_instance(instance, root) for instance in selected}
