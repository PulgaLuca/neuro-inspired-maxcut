# Max-Cut Lab - YAML workflow

## Installazione

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e .
```

Per usare il notebook di analisi, installa anche le dipendenze opzionali:

```bash
pip install -e ".[analysis]"
```

## Comandi

```bash
neuro-inspired-maxcut validate --config configs/benchmark_cpu.yaml
neuro-inspired-maxcut plan --config configs/benchmark_cpu.yaml
neuro-inspired-maxcut run --config configs/benchmark_cpu.yaml
```

In alternativa: `python -m cli run --config configs/benchmark_cpu.yaml`.

Il notebook [`notebooks/analisi_run.ipynb`](./notebooks/analisi_run.ipynb) permette di
analizzare energy, quality e timing di una cartella di run o di tutti i risultati
presenti sotto `results/`.

## Risultati

Ogni lancio crea una cartella `results/<timestamp>_<nome>/` con la configurazione
copiata, `summary.csv`, `results.jsonl`, metadati d'ambiente e una sottocartella
per ogni run con `params.yaml`, `result.json`, `graph.json`, `graph.png`,
`solution.json` e `run.log`.
