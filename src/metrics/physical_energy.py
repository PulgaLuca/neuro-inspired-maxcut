import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


Snapshot = tuple[int, ...]


class _EnergyMeasurementUnavailable(Exception):
    pass


class _EnergyCounter(Protocol):
    def read(self) -> Snapshot: ...

    def difference_joules(self, start: Snapshot, end: Snapshot) -> float: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _RaplPackageCounter:
    energy_paths: tuple[Path, ...]
    max_ranges_uj: tuple[int, ...]

    def read(self) -> Snapshot:
        try:
            values = tuple(int(path.read_text(encoding="utf-8")) for path in self.energy_paths)
        except (OSError, ValueError) as exc:
            raise _EnergyMeasurementUnavailable(
                f"Unable to read Linux RAPL CPU counters: {exc}"
            ) from exc
        return values

    def difference_joules(self, start: Snapshot, end: Snapshot) -> float:
        if len(start) != len(self.max_ranges_uj) or len(end) != len(start):
            raise _EnergyMeasurementUnavailable("The set of RAPL CPU counters changed during the run.")

        difference_uj = 0
        for start_uj, end_uj, max_range_uj in zip(
            start, end, self.max_ranges_uj, strict=True
        ):
            if end_uj >= start_uj:
                difference_uj += end_uj - start_uj
            else:
                difference_uj += max_range_uj - start_uj + end_uj
        return difference_uj / 1_000_000

    def close(self) -> None:
        return None


@dataclass
class _NvmlDeviceCounter:
    module: Any
    handle: Any
    initialized: bool = True

    def read(self) -> Snapshot:
        try:
            energy_mj = self.module.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except self.module.NVMLError as exc:
            raise _EnergyMeasurementUnavailable(
                f"Unable to read NVIDIA NVML GPU energy counter: {exc}"
            ) from exc
        return (int(energy_mj),)

    def difference_joules(self, start: Snapshot, end: Snapshot) -> float:
        if len(start) != 1 or len(end) != 1:
            raise _EnergyMeasurementUnavailable("The NVML GPU energy counter returned invalid data.")
        difference_mj = end[0] - start[0]
        if difference_mj < 0:
            raise _EnergyMeasurementUnavailable("The NVML GPU energy counter decreased during the run.")
        return difference_mj / 1_000

    def close(self) -> None:
        if self.initialized:
            self.module.nvmlShutdown()
            self.initialized = False


@dataclass
class PhysicalEnergyMeter:
    """Measures hardware energy deltas when a supported device counter is available."""

    method: str
    scope: str
    _counter: _EnergyCounter | None
    _start: Snapshot | None
    _error: str | None = None
    _finished: bool = False

    @classmethod
    def start(cls, backend: str, params: dict | None = None) -> "PhysicalEnergyMeter":
        params = params or {}
        if backend == "cpu":
            method = "linux_powercap_rapl_package_energy_counter"
            scope = "cpu_package"
            try:
                counter = _rapl_package_counter()
            except _EnergyMeasurementUnavailable as exc:
                return cls(method, scope, None, None, str(exc))
        elif backend == "gpu":
            method = "nvidia_nvml_total_device_energy_counter"
            scope = "gpu_device"
            try:
                counter = _nvml_device_counter(params)
            except _EnergyMeasurementUnavailable as exc:
                return cls(method, scope, None, None, str(exc))
        else:
            return cls(
                "no_backend_energy_provider",
                "device",
                None,
                None,
                f"No physical energy provider is configured for backend {backend!r}.",
            )

        try:
            start = counter.read()
        except _EnergyMeasurementUnavailable as exc:
            counter.close()
            return cls(method, scope, None, None, str(exc))
        return cls(method, scope, counter, start)

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("Physical energy measurement has already been finished.")
        self._finished = True

        if self._counter is None or self._start is None:
            return self._result(None, "unavailable", self._error)

        try:
            end = self._counter.read()
            joules = self._counter.difference_joules(self._start, end)
        except _EnergyMeasurementUnavailable as exc:
            return self._result(None, "unavailable", str(exc))
        finally:
            self._counter.close()

        return self._result(joules, "measured", None)

    def _result(
        self, joules: float | None, status: str, error: str | None
    ) -> dict[str, Any]:
        return {
            "joules": joules,
            "unit": "J",
            "status": status,
            "method": self.method,
            "scope": self.scope,
            "error": error,
        }


def _rapl_package_counter(
    powercap_root: Path = Path("/sys/class/powercap"),
) -> _RaplPackageCounter:
    if sys.platform != "linux":
        raise _EnergyMeasurementUnavailable(
            "CPU energy measurement is currently supported through Linux powercap RAPL only."
        )

    try:
        package_zones = []
        for zone in sorted(powercap_root.glob("*/")):
            name_file = zone / "name"
            energy_file = zone / "energy_uj"
            range_file = zone / "max_energy_range_uj"
            if not energy_file.is_file() or not range_file.is_file():
                continue
            name = name_file.read_text(encoding="utf-8").strip().lower()
            if "package" not in name:
                continue
            max_range_uj = int(range_file.read_text(encoding="utf-8"))
            if max_range_uj <= 0:
                raise ValueError(f"Invalid RAPL counter range in {range_file}.")
            package_zones.append((energy_file, max_range_uj))
    except (OSError, ValueError) as exc:
        raise _EnergyMeasurementUnavailable(
            f"Unable to discover Linux RAPL CPU package counters: {exc}"
        ) from exc

    if not package_zones:
        raise _EnergyMeasurementUnavailable(
            f"No Linux RAPL CPU package counters found under {powercap_root}."
        )

    return _RaplPackageCounter(
        energy_paths=tuple(path for path, _ in package_zones),
        max_ranges_uj=tuple(max_range for _, max_range in package_zones),
    )


def _nvml_device_counter(params: dict) -> _NvmlDeviceCounter:
    device_index = params.get("device_index", 0)
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise _EnergyMeasurementUnavailable(
            "GPU backend params.device_index must be a non-negative integer."
        )

    try:
        import pynvml
    except ImportError as exc:
        raise _EnergyMeasurementUnavailable(
            "NVIDIA GPU energy measurement requires the optional 'energy' dependencies."
        ) from exc

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_index >= device_count:
            raise _EnergyMeasurementUnavailable(
                f"NVIDIA GPU device_index {device_index} is unavailable; "
                f"NVML detected {device_count} device(s)."
            )
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        if not hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
            raise _EnergyMeasurementUnavailable(
                "The installed NVIDIA driver does not expose a total GPU energy counter."
            )
        return _NvmlDeviceCounter(pynvml, handle)
    except pynvml.NVMLError as exc:
        raise _EnergyMeasurementUnavailable(
            f"Unable to initialize NVIDIA NVML GPU energy measurement: {exc}"
        ) from exc
    except _EnergyMeasurementUnavailable:
        pynvml.nvmlShutdown()
        raise
