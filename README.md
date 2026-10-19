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

## Benchmark GSET

Le istanze GSET sono lette dalla cartella `istances/` in formato
`<numero_nodi> <numero_archi>` seguito da righe `<u> <v> <peso>`.
Gli indici dei nodi nel file, a partire da 1, vengono convertiti internamente
in indici a partire da 0. I grafi sono caricati in forma sparsa, così anche le
istanze grandi non richiedono una matrice densa; i pesi originali, inclusi
quelli negativi, sono mantenuti.

La configurazione di esempio usa due istanze:

```bash
neuro-inspired-maxcut validate --config configs/benchmark_gset.yaml
neuro-inspired-maxcut plan --config configs/benchmark_gset.yaml
neuro-inspired-maxcut run --config configs/benchmark_gset.yaml
```

In `graphs`, impostare `instances: "G1"` per una sola istanza,
`instances: ["G1", "G2"]` per una selezione o `instances: all` per tutti i file
GSET trovati. È possibile specificare un'altra cartella con `path`; i percorsi
relativi sono risolti dalla radice del repository. Per istanze grandi si
consiglia di lasciare `save_graph_image: false`, poiché il rendering del grafo
può essere costoso.

Il notebook [`notebooks/analisi_run.ipynb`](./notebooks/analisi_run.ipynb) permette di
analizzare energy, quality e timing di una cartella di run o di tutti i risultati
presenti sotto `results/`.

La metrica `energy` del notebook è l'energia dell'obiettivo Max-Cut (`-cut_value`),
non una misura fisica dei consumi del dispositivo: la raccolta dei consumi
hardware non è ancora implementata.

## Risultati

Ogni lancio crea una cartella `results/<timestamp>_<nome>/` con la configurazione
copiata, `summary.csv`, `results.jsonl`, metadati d'ambiente e una sottocartella
per ogni run con `params.yaml`, `result.json`, `graph.json`, eventualmente
`graph.png`, `solution.json` e `run.log`. I record GSET riportano anche
`graph_type`, `instance` e `source_file`; il formato dei risultati rimane lo
stesso per confrontarli con Erdos-Renyi e fra backend diversi.
