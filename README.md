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

Per creare rapidamente configurazioni personalizzate, consulta la
[guida alle configurazioni](./configs/configs-helper.md), con esempi per grafi,
algoritmi, backend e indicazioni su cosa è già eseguibile.

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
non il consumo fisico. Ogni nuovo run include `physical_energy` in `result.json`
e `results.jsonl`; `summary.csv` espone i campi come `physical_energy_joules`,
`physical_energy_status`, `physical_energy_method`, `physical_energy_scope` e
`physical_energy_error`. Il valore è in joule solo quando `status` è
`"measured"`; se un contatore non è disponibile, `joules` è `null` e il motivo
è registrato in `error`.

La misura CPU usa i contatori package RAPL esposti da Linux powercap. La misura
GPU usa il contatore energetico totale di una GPU NVIDIA tramite NVML; installa
il supporto opzionale con `pip install -e ".[energy]"` e, se necessario,
seleziona la scheda con `backends[].params.device_index` (default `0`). Sono
misure dell'intero package CPU o dispositivo GPU durante il run, non
un'attribuzione al singolo processo: includono quindi l'attività concorrente
sullo stesso dispositivo. Su Windows, GPU non NVIDIA o host senza contatore
supportato la misura risulta esplicitamente non disponibile. I backend
neuromorfico e quantum richiedono ancora un provider specifico e per ora
riportano `status: "unavailable"`. Il notebook include il consumo misurato
nell'analisi per run, nella media per gruppo e nei grafici; le misure mancanti
non vengono sostituite con stime.

## Risultati

Ogni lancio crea una cartella `results/<timestamp>_<nome>/` con la configurazione
copiata, `summary.csv`, `results.jsonl`, metadati d'ambiente e una sottocartella
per ogni run con `params.yaml`, `result.json`, `graph.json`, eventualmente
`graph.png`, `solution.json` e `run.log`. I record GSET riportano anche
`graph_type`, `instance` e `source_file`; il formato dei risultati rimane lo
stesso per confrontarli con Erdos-Renyi e fra backend diversi.
