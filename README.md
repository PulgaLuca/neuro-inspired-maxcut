# Max-Cut Lab - YAML workflow

## Architettura e stato implementato

Il dominio è in `src/problems` e `src/contracts.py`; generatori e lettore GSET
sono in `src/graphs`; gli algoritmi CPU effettivi sono `exact`, `local_search`,
`simulated_annealing`, `hopfield` e `boltzmann`. La registry in `src/algorithms/registry.py` associa
ogni algoritmo alla factory e ai backend dichiarati; `src/backends/registry.py`
dichiara quali provider hardware sono davvero disponibili. `src/experiments` pianifica
ed esegue il prodotto cartesiano e `src/storage` mantiene il formato storico
`result.json`, `results.jsonl` e `summary.csv`, aggiungendo `schema_version: 1`
e metadati riproducibili.

`exact`, `local_search` e `simulated_annealing` hanno percorsi CPU e GPU.
L'esecuzione GPU usa CuPy/CUDA, installabile con `pip install -e "[gpu]"` su
un host compatibile; `plan` controlla che CuPy rilevi una scheda prima di
dichiarare disponibile la combinazione. Il percorso GPU di Local Search usa
best-improvement (valuta in parallelo tutti i flip), mentre quello CPU usa
first-improvement. La GPU di Simulated Annealing esegue catene indipendenti
in parallelo (`gpu_replicas`, default 128); Exact enumera le soluzioni in
batch (`gpu_batch_size`, default 4096). Queste strategie sono registrate nei
metadati del risultato e possono produrre qualità e tempi diversi dalla CPU.

Hopfield è disponibile solo su CPU: usa dinamica asincrona a soglia zero con
ordine casuale dei neuroni, mantiene lo spin in caso di pareggio e termina a
convergenza o al limite configurato. L'esempio `configs/benchmark_heuristics_cpu.yaml`
confronta Hopfield, Boltzmann e Local Search a parità di grafo, seed di
ripetizione e backend. Boltzmann usa campionamento Gibbs a temperatura fissa e
restituisce la migliore soluzione incontrata; è distinto da Simulated Annealing,
che usa proposte Metropolis e raffreddamento. Entrambi sono disponibili solo su
CPU. SNN (`spiking_neural_network`) è disponibile come simulatore software CPU
di neuroni LIF accoppiati: è confrontabile per qualità e tempo, ma non equivale
a esecuzione neuromorfica. Il risultato riporta `provider: cpu_snn_simulator`
e `neuromorphic_hardware_used: false`. Quantum e neuromorfico restano
destinazioni senza provider; QUBO non è implementato e non ci sono provider GPU
remoti. Per aggiungere un algoritmo o un adapter backend, implementare il
contratto `solve` e registrare una factory e i backend supportati in
`algorithms/registry.py`, poi dichiarare il provider in `backends/registry.py`.
I contatori di energia sono separati: CPU usa RAPL se esposto dal sistema e
GPU usa NVML se supportato dal driver e installato l'extra `energy`. In ogni
altro caso joule resta `null` con stato `unavailable`. La misura CPU durante
SNN riguarda il package CPU, non un dispositivo neuromorfico. Per registrare
energia neuromorfica servono un provider e un contatore fisico specifici; il
framework non stima i valori mancanti.

QAOA (`qaoa`) supporta due percorsi distinti. `quantum_simulator` è una
simulazione classica CPU con stato vettoriale esponenziale (`2^n` ampiezze),
adatta a grafi piccoli. `quantum` usa Qiskit Runtime SamplerV2 per inviare
circuiti a un QPU IBM reale, selezionato esplicitamente o scelto fra quelli
disponibili. Per quest'ultimo installare `pip install -e ".[quantum]"` e
fornire il token via `IBM_QUANTUM_TOKEN`; il file locale escluso da Git
`src/apikey.json` è letto come fallback per la chiave `apikey`. Le credenziali
non vengono incluse nei risultati. I job reali sono stocastici e possono
richiedere attesa/quote del servizio. Il contatore energia per il QPU IBM non
è integrato: i joule restano `null`/`unavailable`, mai stimati. Esempi:
`configs/benchmark_qaoa_simulator.yaml` e `configs/benchmark_qaoa_ibm.yaml`.
Vedi la [guida di configurazione](./configs/configs-helper.md).

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

Per confrontare le tre implementazioni su CPU e GPU con un grafo piccolo,
consulta `configs/benchmark_cpu_gpu.yaml`. Le run GPU sono eseguite solo se il
provider CuPy/CUDA e il dispositivo selezionato risultano disponibili.

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
presenti sotto `results/`. Include grafici run-level tempo/qualità con mediana e
intervallo interquartile, energia fisica misurata/qualità, tempo/energia,
qualità e scalabilità rispetto ai nodi con pannelli per famiglia di grafi,
rapporti normalizzati rispetto a Exact CPU (o al best observed quando Exact
manca), speedup rispetto alla CPU dello stesso algoritmo e frontiera Pareto
tempo/energia/qualità. Le funzioni riutilizzabili sono in `src/analysis/plots.py`.
I grafici energetici e Pareto usano solo run completate con `status: measured`;
quando non ci sono misure, il notebook lo segnala senza imputare valori.
Impostando `FIGURES_FOLDER` nella nuova cella notebook si esportano i grafici
come PNG e PDF per la relazione.

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
