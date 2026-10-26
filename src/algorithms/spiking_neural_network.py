"""Paired leaky integrate-and-fire simulation for weighted Max-Cut."""
from __future__ import annotations

import time

import numpy as np

from algorithms.base import Algorithm
from contracts import SolveResult
from problems.maxcut import MaxCutProblem


class SpikingNeuralNetwork(Algorithm):
    """CPU simulation with two competing LIF neurons for each Max-Cut spin.

    Positive and negative populations encode the two spin values. Recurrent
    input uses ``-W``: positive input to the positive-spin neuron corresponds
    to a local field that favors a larger cut. A spike selects that spin and
    resets/refracts the competing pair. This is a software simulator, not a
    neuromorphic hardware provider.
    """

    name = "spiking_neural_network"

    def __init__(
        self,
        time_steps: int = 1000,
        threshold: float = 1.0,
        membrane_decay: float = 0.2,
        synaptic_scale: float = 0.1,
        refractory_steps: int = 1,
    ) -> None:
        if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps < 1:
            raise ValueError("time_steps deve essere un intero >= 1.")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("threshold deve essere finita e > 0.")
        if (
            isinstance(membrane_decay, bool)
            or not isinstance(membrane_decay, (int, float))
            or not np.isfinite(membrane_decay)
            or not 0 <= membrane_decay < 1
        ):
            raise ValueError("membrane_decay deve essere finita e compresa tra 0 e 1.")
        if (
            isinstance(synaptic_scale, bool)
            or not isinstance(synaptic_scale, (int, float))
            or not np.isfinite(synaptic_scale)
            or synaptic_scale <= 0
        ):
            raise ValueError("synaptic_scale deve essere finita e > 0.")
        if (
            isinstance(refractory_steps, bool)
            or not isinstance(refractory_steps, int)
            or refractory_steps < 0
        ):
            raise ValueError("refractory_steps deve essere un intero >= 0.")

        self.time_steps = time_steps
        self.threshold = float(threshold)
        self.membrane_decay = float(membrane_decay)
        self.synaptic_scale = float(synaptic_scale)
        self.refractory_steps = refractory_steps

    def solve(
        self,
        graph: MaxCutProblem,
        hardware: str,
        seed: int = 0,
        time_limit_seconds: float | None = None,
        **params,
    ) -> SolveResult:
        if hardware != "cpu":
            raise ValueError(f"Hardware not supported by {self.name}: {hardware}")

        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        node_count = graph.n_variables
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=node_count)
        best = spins.copy()
        current_value = float(graph.evaluate(spins))
        best_value = current_value
        best_time_step = 0

        positive_voltage = np.zeros(node_count, dtype=np.float64)
        negative_voltage = np.zeros(node_count, dtype=np.float64)
        refractory = np.zeros(node_count, dtype=np.int64)
        time_steps_completed = 0
        neuron_updates = 0
        positive_spikes = 0
        negative_spikes = 0
        timed_out = False

        for time_step in range(1, self.time_steps + 1):
            if (
                time_limit_seconds is not None
                and time.perf_counter() - start >= time_limit_seconds
            ):
                timed_out = True
                break

            for node in rng.permutation(node_count):
                if (
                    time_limit_seconds is not None
                    and time.perf_counter() - start >= time_limit_seconds
                ):
                    timed_out = True
                    break

                node_index = int(node)
                neuron_updates += 1
                if refractory[node_index] > 0:
                    refractory[node_index] -= 1
                    continue

                local_field = graph.flip_delta(spins, node_index) * int(spins[node_index])
                positive_drive = max(-local_field, 0.0)
                negative_drive = max(local_field, 0.0)
                retention = 1.0 - self.membrane_decay
                positive_voltage[node_index] = (
                    retention * positive_voltage[node_index]
                    + self.synaptic_scale * positive_drive
                )
                negative_voltage[node_index] = (
                    retention * negative_voltage[node_index]
                    + self.synaptic_scale * negative_drive
                )

                positive_fired = positive_voltage[node_index] >= self.threshold
                negative_fired = negative_voltage[node_index] >= self.threshold
                if not positive_fired and not negative_fired:
                    continue

                positive_spikes += int(positive_fired)
                negative_spikes += int(negative_fired)
                if positive_fired and negative_fired:
                    if positive_voltage[node_index] > negative_voltage[node_index]:
                        new_spin = 1
                    elif negative_voltage[node_index] > positive_voltage[node_index]:
                        new_spin = -1
                    else:
                        new_spin = int(spins[node_index])
                else:
                    new_spin = 1 if positive_fired else -1

                positive_voltage[node_index] = 0.0
                negative_voltage[node_index] = 0.0
                refractory[node_index] = self.refractory_steps

                if new_spin != spins[node_index]:
                    delta = graph.flip_delta(spins, node_index)
                    spins[node_index] = new_spin
                    current_value += delta
                    if current_value > best_value:
                        best_value = current_value
                        best = spins.copy()
                        best_time_step = time_step

            if timed_out:
                break
            time_steps_completed = time_step

        elapsed = time.perf_counter() - start
        total_spikes = positive_spikes + negative_spikes
        firing_rate = (
            total_spikes / (2 * node_count * time_steps_completed)
            if time_steps_completed
            else 0.0
        )
        best_value = float(graph.evaluate(best))
        return SolveResult(
            self.name,
            hardware,
            best,
            best_value,
            elapsed,
            "time_limit" if timed_out else "completed",
            seed,
            {
                "model": "paired_competitive_lif",
                "dynamics": "asynchronous_discrete_lif",
                "time_steps_requested": self.time_steps,
                "time_steps_completed": time_steps_completed,
                "neuron_updates": neuron_updates,
                "positive_spikes": positive_spikes,
                "negative_spikes": negative_spikes,
                "total_spikes": total_spikes,
                "firing_rate": firing_rate,
                "best_time_step": best_time_step,
                "best_objective_value": best_value,
                "final_chain_objective_value": float(graph.evaluate(spins)),
                "threshold": self.threshold,
                "membrane_decay": self.membrane_decay,
                "synaptic_scale": self.synaptic_scale,
                "refractory_steps": self.refractory_steps,
                "provider": "cpu_snn_simulator",
                "device": "CPU simulation",
                "neuromorphic_hardware_used": False,
            },
        )
