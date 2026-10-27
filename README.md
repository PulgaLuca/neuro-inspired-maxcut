# Neuro-Inspired Max-Cut

Explore how classical and neuro-inspired optimization methods solve the
[Max-Cut problem](https://en.wikipedia.org/wiki/Maximum_cut): split a graph's
vertices into two groups to maximize the total weight of edges crossing the
partition.

This Python framework runs reproducible, YAML-configured benchmarks on
Erdos–Renyi graphs and GSET instances. Compare algorithms on shared graphs,
inspect their plans before execution, and find structured results under
`results/`.

## Get started

Requires Python 3.11–3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Validate, preview, and run a CPU benchmark:

```powershell
neuro-inspired-maxcut validate --config configs\benchmark_heuristics_cpu.yaml
neuro-inspired-maxcut plan --config configs\benchmark_heuristics_cpu.yaml
neuro-inspired-maxcut run --config configs\benchmark_heuristics_cpu.yaml
```

To try GSET data instead, use `configs\benchmark_gset.yaml`. The bundled
instances live in `istances\`; the configuration guide explains how to select
instances and customize graph, algorithm, and backend settings.

## What’s included

- **Algorithms:** exact search, local search, simulated annealing, Hopfield
  networks, Boltzmann machines, and a software spiking-neural-network simulator.
- **Backends:** CPU support, plus GPU implementations for selected algorithms.
  GPU runs require a compatible CUDA setup; install with
  `python -m pip install -e ".[gpu]"`.
- **Quantum:** QAOA is available as a classical simulator or through the
  optional IBM Quantum provider. The simulator is not quantum hardware, and
  hardware runs require the `quantum` extra and IBM credentials.

Not every algorithm supports every backend. Use `plan` to check availability
before running; unsupported hardware integrations are reported rather than
silently simulated.

Each experiment saves its configuration, run records, and summary files in
`results/`. For configuration examples and details, see the
[configuration guide](configs/configs-helper.en.md).
