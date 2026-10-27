# Configuration Quick Guide

This guide explains how to create YAML files for benchmarking combinations of
graphs, algorithms, and hardware. Configuration files are declarative: adding
a name to YAML does not implement the corresponding algorithm or backend.
Before starting a benchmark, distinguish configurations that can run today
from examples intended for future integrations.

## Commands

From PowerShell, in the repository root:

```powershell
neuro-inspired-maxcut validate --config configs\my_config.yaml
neuro-inspired-maxcut plan --config configs\my_config.yaml
neuro-inspired-maxcut run --config configs\my_config.yaml
```

`validate` checks the configuration structure and key values; `plan` shows how
many runs will be created; `run` executes the benchmark and saves results under
the directory configured by `experiment.output.root`. For a first run, always
use `validate` and `plan` before `run`.

## Common structure

```yaml
experiment:
  name: "my_experiment"
  seed: 99
  repetitions: 3
  output:
    root: "results"
    save_graph_image: false
    save_solution: true
    save_logs: true

execution:
  time_limit_seconds: 60
  fail_fast: false

graphs:
  - name: "gset"
    type: "gset"
    instances: ["G1", "G2"]

algorithms:
  - name: "simulated_annealing"
    params:
      steps: 10000
      initial_temperature: 10.0
      final_temperature: 0.01

backends:
  - name: "cpu"
    params: {}
```

### How the fields combine

- Every entry in `graphs`, `algorithms`, and `backends` is combined with the
  others. For example, 2 graphs, 2 algorithms, 2 backends, and 3 repetitions
  schedule `2 x 2 x 2 x 3 = 24` runs.
- `experiment.seed` makes experiments reproducible; `repetitions` must be at
  least 1. Each graph is generated once and reused throughout the plan. The
  algorithm seed is derived from the base seed, graph identifier, and
  repetition: algorithms and backends being compared receive the same seed
  for the same graph and repetition.
- `execution.time_limit_seconds` is the limit for each run, not the whole
  experiment. With `fail_fast: false`, a failed run is recorded and the runner
  continues with the remaining runs.
- `save_graph_image: false` is recommended for large graphs.
- For fair comparisons, keep the graph, seed, repetitions, and algorithm
  parameters fixed, changing only the backend or the variable you want to
  measure.

## Graph sources

### GSET

Files are searched for in the repository's `istances/` directory. Select one
or more instances by name:

```yaml
graphs:
  - name: "gset"
    type: "gset"
    instances: "G1"             # A single instance
  - name: "another_gset"
    type: "gset"
    instances: ["G2", "G3"]     # A selection
```

To include every instance found:

```yaml
graphs:
  - name: "all_gset"
    type: "gset"
    instances: all
```

You can specify a different directory with `path`; relative paths are resolved
from the repository root:

```yaml
graphs:
  - name: "my_gset"
    type: "gset"
    path: 'data\gset'
    instances: ["G1"]
```

For GSET, `sizes` is not required: the number of nodes is read from the file.
Instances may contain negative weights. Use an algorithm that directly
maximizes the Max-Cut objective over signed weights.

### Erdos-Renyi

```yaml
graphs:
  - name: "er_sparse"
    type: "erdos_renyi"
    sizes: [100, 200]
    density: 0.05
    weighted: false
  - name: "er_weighted"
    type: "erdos_renyi"
    sizes: [100]
    density: 0.2
    weighted: true
```

`density` must be between 0 and 1. `weighted` enables random positive edge
weights; the same configuration values and seed recreate the same graph for
every algorithm and backend in the plan.

## Algorithms: support and configuration

| YAML name | Repository support | Known parameters |
|---|---|---|
| `simulated_annealing` | CPU; GPU with CuPy/CUDA and parallel chains | `steps`, temperatures, `gpu_replicas` |
| `exact` | CPU and GPU in batches, up to `max_nodes` | `max_nodes`, `gpu_batch_size` |
| `local_search` | CPU first-improvement; GPU best-improvement | `max_iterations` |
| `hopfield` | CPU; asynchronous dynamics, random neuron order | `max_iterations` |
| `boltzmann` | CPU; fixed-temperature Gibbs/heat-bath sampling | `steps`, `temperature` |
| `spiking_neural_network` (SNN) | Software LIF simulator on CPU; real neuromorphic provider unavailable | `time_steps`, `threshold`, `membrane_decay`, `synaptic_scale`, `refractory_steps` |
| `qubo` solvers | Not implemented | — |

Validation accepts only registered, implemented algorithms. The plan explicitly
marks unsupported hardware combinations; during `run`, these are persisted as
errors without falling back to another backend.

### Simulated annealing: runnable example

```yaml
algorithms:
  - name: "simulated_annealing"
    params:
      steps: 100000
      initial_temperature: 10.0
      final_temperature: 0.01
```

`steps` must be at least 1; both temperatures must be positive, and
`final_temperature` must not exceed `initial_temperature`. The GPU variant uses
multiple independent chains; `gpu_replicas` sets the number of chains to run in
parallel. CPU and GPU use the same temperature scale but do not follow the same
trajectory.

### Exact: example for small instances

```yaml
algorithms:
  - name: "exact"
    params:
      max_nodes: 20
```

The algorithm enumerates solutions, so its cost grows exponentially:
`max_nodes` is a safety limit, not a time estimate. CPU and GPU support only
instances within that limit; on GPU, `gpu_batch_size` controls the number of
solutions processed per batch. Do not use Exact on large GSET instances.

### CPU SNN and neuromorphic provider

The available SNN implementation is a CPU software simulator with two
competitive LIF neurons per node, encoding spins `-1` and `+1`. It produces a
Max-Cut solution verified like those from other algorithms, but it does not
use neuromorphic hardware. Runtime and spike counts describe the CPU
simulation; they do not measure or predict the performance of a neuromorphic
device.

```yaml
algorithms:
  - name: "spiking_neural_network"
    params:
      time_steps: 1000
      threshold: 1.0
      membrane_decay: 0.2
      synaptic_scale: 0.1
      refractory_steps: 1
```

`time_steps` is the maximum number of steps, `threshold` is the spike
threshold, `membrane_decay` is the potential lost per step, `synaptic_scale`
sets the input scale, and `refractory_steps` is the refractory period. The
seed controls the initial state and update order. The CPU simulation can be
included in the comparison in `configs/benchmark_heuristics_cpu.yaml`.

The `neuromorphic` backend is intentionally unavailable until a hardware
provider is registered. An integration must provide an algorithm adapter for
`spiking_neural_network` through `register_backend_implementation(...)` and
declare the provider available with `register_backend(BackendCapability(...))`
before planning runs. A dedicated energy counter is registered separately
with `register_physical_energy_provider(...)`. The application must explicitly
load the provider module; the framework does not simulate hardware
automatically. Without a physical counter, energy remains `null` with status
`unavailable`.

### CPU Hopfield

```yaml
algorithms:
  - name: "hopfield"
    params:
      max_iterations: 100000
```

Hopfield uses asynchronous updates in random neuron order. The network uses
couplings `-W`, where `W` is the graph's weight matrix: every update that
changes a spin strictly increases the cut; in a tie, the spin stays unchanged.
`max_iterations` limits neuron evaluations. The result records updates, flips,
full passes, and convergence. This variant is CPU-only and uses the same
normalized result contract as the other algorithms. For a direct comparison
with Local Search, see `configs/benchmark_heuristics_cpu.yaml`.

### CPU Boltzmann machine

```yaml
algorithms:
  - name: "boltzmann"
    params:
      steps: 50000
      temperature: 1.0
```

Boltzmann uses Gibbs (heat-bath) updates: at each step, it selects a neuron and
resamples it from the conditional distribution at the fixed `temperature`. It
returns the state with the best cut found during the chain. It differs from
Simulated Annealing because it uses conditional probabilities rather than
Metropolis proposals with cooling. `steps` counts neuron updates; the seed and
parameters are saved in every run. It is currently CPU-only. The example
`configs/benchmark_heuristics_cpu.yaml` compares Boltzmann with Hopfield and
Local Search.

You can list multiple algorithms in the same file when they are available:
the plan will run each one on every selected graph and backend.

## Hardware backends

The configuration field is named `backends`:

```yaml
backends:
  - name: "cpu"
    params: {}
```

CPU is always available. GPU variants of Exact, Local Search, and Simulated
Annealing require CuPy and a usable CUDA device. Install
`pip install -e ".[gpu]"` (the extra is based on CUDA 12) and select the device:

```yaml
backends:
  - name: "gpu"
    params:
      device_index: 0
```

`plan` checks the provider and device availability. If CuPy or CUDA is missing,
the combination is marked unsupported and `run` persists an error; there is no
CPU fallback. GPU Local Search uses best-improvement instead of CPU
first-improvement. GPU Simulated Annealing uses multiple independent chains
(`gpu_replicas`), while Exact enumerates in batches (`gpu_batch_size`). These
strategies are recorded in the metadata to make comparisons easier to
interpret. Quantum and neuromorphic hardware still require dedicated
providers; `quantum_simulator` is instead a separate classical backend for
QAOA.

```yaml
# Optional GPU and hardware providers:
backends:
  - name: "gpu"
    params:
      device: 0
  - name: "neuromorphic"
    params:
      device: "device_name"
  - name: "quantum"
    params:
      provider: "provider_name"
      shots: 1000
```

The backend `params` keys above are examples of possible options, not
options implemented or validated by the project. Define parameter names and
semantics when integrating the relevant SDK or device. Likewise, listing a
backend alone does not make an algorithm compatible with it: the algorithm
must implement the path for that hardware.

### QAOA: simulator and quantum provider

`qaoa` supports both the `quantum_simulator` backend and IBM Quantum Runtime.
The simulator runs on CPU using NumPy and a state vector of size `2^n`, so it
is useful only as a reference for small graphs. The IBM implementation sends
QAOA circuits to `SamplerV2` on a real QPU; each run is stochastic and requires
access, quota, and queue time. `max_nodes` is a software limit for the graph.
Backend selection is explicit: `quantum` does not fall back to the simulator.

```yaml
algorithms:
  - name: "qaoa"
    params:
      depth: 1
      optimizer_iterations: 30
      optimizer_restarts: 2
      max_nodes: 16
backends:
  - name: "quantum_simulator"
    params: {}
```

Results identify `provider: numpy_statevector_simulator`,
`device: CPU state-vector simulation`, and `quantum_hardware_used: false`.
When available, the RAPL counter measures the CPU package used by the
simulator, not quantum hardware.

For IBM, install `pip install -e ".[quantum]"`. Authentication first uses
`IBM_QUANTUM_TOKEN`; if it is not set, the framework reads `apikey` from the
Git-ignored local file `src/apikey.json`. Alternatively, specify an external
credentials file with `IBM_QUANTUM_TOKEN_FILE`. Do not put tokens in YAML.
The default channel is `ibm_quantum_platform`; you can specify a particular
backend or let the provider select the least busy one with the required number
of qubits:

```yaml
backends:
  - name: "quantum"
    params:
      channel: "ibm_quantum_platform"
      # backend_name: "ibm_backend_name"  # Optional
      # instance: "instance-id"            # Optional
```

The plan checks for the SDK and local token. Connection, authorization, and
QPU availability are checked during `run`; authentication or service errors
become failed runs with diagnostics, without switching to the simulator.
Metadata includes the IBM backend, job identifiers, depth, and shots, but not
tokens. QPU energy counters are unavailable, so `physical_energy.joules`
remains null.

`configs/benchmark_qaoa_simulator.yaml` is the simulated QAOA example, and
`configs/benchmark_qaoa_ibm.yaml` is the IBM QPU example. To compare the same
graphs with CPU heuristics, use `configs/benchmark_qaoa_cpu.yaml`; these are
separate configurations because the framework plans the Cartesian product of
algorithms and backends and must not pair QAOA with the CPU backend or
heuristics with the quantum simulator.

## Example scenarios

### Compare two CPU algorithms on a small graph

This example runs with the current code because it uses only
`simulated_annealing` and `exact` on a 10-node Erdos-Renyi graph:

```yaml
experiment:
  name: "small_cpu_comparison"
  seed: 99
  repetitions: 5
  output:
    root: "results"
    save_graph_image: false
    save_solution: true
    save_logs: true
execution:
  time_limit_seconds: 60
  fail_fast: false
graphs:
  - name: "small_er"
    type: "erdos_renyi"
    sizes: [10]
    density: 0.3
    weighted: false
algorithms:
  - name: "exact"
    params:
      max_nodes: 20
  - name: "simulated_annealing"
    params:
      steps: 10000
      initial_temperature: 10.0
      final_temperature: 0.01
backends:
  - name: "cpu"
    params: {}
```

If the instance exceeds `max_nodes`, validation reports it for `exact`; for
larger graphs, remove `exact` and use implemented heuristic algorithms.

### Large matrix (illustrative fragment)

The following fragment illustrates a mixed comparison, but is not a complete
runnable configuration: Hopfield and SNN support only CPU, while the other
three implementations support CPU and GPU when CuPy/CUDA are available.
Combinations without a provider are reported by the plan. The neuromorphic
backend requires a dedicated adapter as described above.

```yaml
algorithms:
  - name: "simulated_annealing"
    params:
      steps: 100000
      initial_temperature: 10.0
      final_temperature: 0.01
  - name: "hopfield"
    params: {}
  - name: "spiking_neural_network"
    params:
      time_steps: 1000
backends:
  - name: "cpu"
    params: {}
  - name: "gpu"
    params: {}
  - name: "neuromorphic"
    params: {}
  - name: "quantum"
    params: {}
```

For a Hopfield/Local Search comparison that can run immediately, use
`configs/benchmark_heuristics_cpu.yaml`; for algorithms with CPU/GPU variants,
see `configs/benchmark_cpu_gpu.yaml`.

## Comparable experiments and results

Each run saves the algorithm, backend, seed, parameters, graph/instance,
solution, objective, runtime, and physical energy measurements in the usual
`result.json`, `results.jsonl`, and `summary.csv` files. In JSON, the
measurement is stored in the `physical_energy` object (`joules`, `unit`,
`status`, `method`, `scope`, `error`); the CSV flattens these into fields such
as `physical_energy_joules`. `joules` is populated only when `status` is
`measured`; if a sensor is unavailable, the value is `null` and `error`
explains why.

The notebook's `energy` metric is `-cut_value`, the mathematical energy of the
Max-Cut objective; it does **not** measure the computer's or device's power
consumption. The notebook distinguishes `physical_energy_joules` and includes
the mean of available measurements in analyses by graph, algorithm, and
backend.

Physical CPU energy is read from package RAPL counters exposed through Linux
powercap. GPU energy is read from the total NVML energy counter of the selected
NVIDIA GPU, configured through `backends[].params.device_index` (default `0`);
install support with `pip install -e ".[energy]"`. Both measure the entire
CPU package or GPU during the run, so concurrent activity is included and the
measurement is not attributed to the process alone. On Windows, without a
compatible counter, on non-NVIDIA GPUs, and for neuromorphic/quantum backends
without a specific provider, the result reports `unavailable` and does not
use estimates.
