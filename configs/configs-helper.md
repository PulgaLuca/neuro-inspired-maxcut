# Guida rapida alle configurazioni

Questa guida aiuta a creare file YAML per provare combinazioni di grafi,
algoritmi e hardware. I file di configurazione sono dichiarativi: aggiungere un
nome a YAML non implementa l'algoritmo o il backend corrispondente. Prima di
avviare un benchmark, distinguere quindi le configurazioni già eseguibili dagli
schemi per esperimenti futuri.

## Comandi

Da PowerShell, nella cartella principale del repository:

```powershell
neuro-inspired-maxcut validate --config configs\mia_configurazione.yaml
neuro-inspired-maxcut plan --config configs\mia_configurazione.yaml
neuro-inspired-maxcut run --config configs\mia_configurazione.yaml
```

`validate` controlla la struttura e i valori principali; `plan` mostra quante
esecuzioni verranno create; `run` esegue il benchmark e salva i risultati sotto
la directory configurata in `experiment.output.root`. Per una prima prova, usare
sempre `validate` e `plan` prima di `run`.

## Struttura comune

```yaml
experiment:
  name: "nome_esperimento"
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

### Come si combinano i campi

- Ogni elemento in `graphs`, `algorithms` e `backends` viene combinato con gli
  altri. Per esempio, 2 grafi, 2 algoritmi, 2 backend e 3 ripetizioni
  pianificano `2 x 2 x 2 x 3 = 24` run.
- `experiment.seed` rende riproducibili gli esperimenti; `repetitions` deve
  essere almeno 1.
- `execution.time_limit_seconds` è il limite per ogni run, non per l'intero
  esperimento. Con `fail_fast: false`, un run fallito viene registrato e il
  runner continua con gli altri.
- `save_graph_image: false` è consigliato per grafi grandi.
- Per confronti confrontabili, mantieni invariati grafo, seed, ripetizioni e
  parametri degli algoritmi, modificando solo il backend o la variabile che
  vuoi misurare.

## Grafi selezionabili

### GSET

I file sono cercati nella cartella `istances/` del repository. Seleziona una o
più istanze per nome:

```yaml
graphs:
  - name: "gset"
    type: "gset"
    instances: "G1"             # Una sola istanza
  - name: "gset_altro"
    type: "gset"
    instances: ["G2", "G3"]     # Una selezione
```

Per includere tutte le istanze trovate:

```yaml
graphs:
  - name: "gset_completo"
    type: "gset"
    instances: all
```

È possibile indicare un'altra cartella con `path`; un percorso relativo viene
risolto dalla radice del repository:

```yaml
graphs:
  - name: "gset_personale"
    type: "gset"
    path: 'dati\gset'
    instances: ["G1"]
```

Per GSET il campo `sizes` non serve: il numero di nodi viene letto dal file.
Le istanze possono avere pesi negativi. Usare un algoritmo che massimizza
direttamente l'obiettivo Max-Cut sui pesi con segno.

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

`density` deve essere compresa tra 0 e 1. `weighted` abilita pesi casuali
positivi; gli stessi valori di configurazione e seed ricreano lo stesso grafo
per ogni algoritmo e backend del piano.

## Algoritmi: stato e configurazione

| Nome YAML | Stato nel repository | Parametri conosciuti |
|---|---|---|
| `simulated_annealing` | Eseguibile su CPU | `steps`, `initial_temperature`, `final_temperature` |
| `exact` | Eseguibile su CPU solo per grafi piccoli | `max_nodes` |
| `local_search` | Non ancora collegato al runner | Da definire con l'implementazione |
| `hopfield` | Non ancora collegato al runner | Da definire con l'implementazione |
| `spiking_neural_network` (SNN) | Non ancora collegato al runner | Da definire con l'implementazione |
| `boltzmann` | Non implementato né registrato | Nome e parametri da concordare con l'implementazione |

Gli algoritmi non ancora collegati possono essere accettati dalla validazione
del nome (quando presenti in `supported_configurations.yaml`), ma la creazione
del piano di esecuzione fallirà perché il runner non sa ancora istanziarli.
Non aggiungere parametri ipotetici sperando che vengano usati: il contratto
`params` dovrà essere definito insieme all'implementazione.

### Simulated annealing: esempio eseguibile

```yaml
algorithms:
  - name: "simulated_annealing"
    params:
      steps: 100000
      initial_temperature: 10.0
      final_temperature: 0.01
```

`steps` deve essere almeno 1; entrambe le temperature devono essere positive e
`final_temperature` non deve superare `initial_temperature`. Il backend CPU è
quello attualmente implementato.

### Exact: esempio per istanze piccole

```yaml
algorithms:
  - name: "exact"
    params:
      max_nodes: 20
```

L'algoritmo enumera le soluzioni, quindi il costo cresce esponenzialmente:
`max_nodes` è un limite di sicurezza, non una stima di tempo. L'implementazione
CPU supporta solo istanze entro tale limite. Il backend GPU non è implementato.
Non usare questo algoritmo sulle istanze GSET grandi.

### Local search, Hopfield, SNN e Boltzmann: scheletro futuro

Queste voci indicano la forma che avrà una configurazione una volta implementati
i rispettivi algoritmi; **non sono run eseguibili oggi**. Prima di usarle,
implementare e registrare l'algoritmo nel runner e definire i parametri ammessi.

```yaml
algorithms:
  - name: "local_search"
    params: {}
  - name: "hopfield"
    params: {}
  - name: "spiking_neural_network"
    params: {}
  # "boltzmann" va aggiunto alle configurazioni supportate e implementato.
  # algorithms:
  #   - name: "boltzmann"
  #     params: {}
```

È possibile elencare più algoritmi nello stesso file quando sono disponibili:
il piano eseguirà ciascuno su ogni grafo e backend selezionato.

## Backend hardware

Nella configurazione il campo si chiama `backends`:

```yaml
backends:
  - name: "cpu"
    params: {}
```

I nomi `cpu`, `gpu`, `neuromorphic` e `quantum` sono attualmente presenti tra
i valori ammessi. Questo non significa che siano tutti utilizzabili: oggi
`simulated_annealing` e `exact` eseguono solo il percorso CPU; il percorso GPU
è incompleto e non esistono ancora adattatori per dispositivi quantum o
neuromorfici. Perciò le configurazioni qui sotto documentano scenari futuri e
non devono essere considerate eseguibili.

```yaml
# Solo dopo aver implementato i backend corrispondenti:
backends:
  - name: "gpu"
    params:
      device: 0
  - name: "neuromorphic"
    params:
      device: "nome_dispositivo"
  - name: "quantum"
    params:
      provider: "nome_provider"
      shots: 1000
```

Le chiavi sotto `params` del backend sono esempi di possibili opzioni, non
opzioni implementate o validate dal progetto. Definire i nomi e i significati
dei parametri quando si integra il relativo SDK o dispositivo. Analogamente,
un backend elencato da solo non rende un algoritmo compatibile con esso:
l'algoritmo deve implementare il percorso per quell'hardware.

## Esempi di scenari

### Confrontare due algoritmi CPU su un grafo piccolo

Questo esempio è eseguibile con il codice attuale perché usa solo
`simulated_annealing` ed `exact` su un grafo Erdos-Renyi di 10 nodi:

```yaml
experiment:
  name: "confronto_cpu_piccolo"
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
  - name: "er_piccolo"
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

Se l'istanza supera `max_nodes`, la validazione la segnala per `exact`; per
grafi più grandi rimuovere `exact` e usare algoritmi euristici già implementati.

### Matrice di algoritmi e dispositivi (schema futuro)

Quando gli algoritmi e i backend saranno implementati, una singola
configurazione potrà descrivere il prodotto cartesiano desiderato:

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
    params: {}
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

L'esempio è uno schema, non un esperimento avviabile con le implementazioni
attuali: alcuni algoritmi non sono collegati e simulated annealing non supporta
ancora GPU, neuromorfico o quantum.

## Esperimenti comparabili e risultati

Ogni run salva algoritmo, backend, seed, parametri, istanza/grafo, soluzione,
obiettivo, tempo e misura dei consumi fisici nei normali file `result.json`,
`results.jsonl` e `summary.csv`. Nel JSON la misura è contenuta nell'oggetto
`physical_energy` (`joules`, `unit`, `status`, `method`, `scope`, `error`); il
CSV appiattisce questi valori con nomi come `physical_energy_joules`.
`joules` è valorizzato solo quando `status` è `measured`; se il sensore manca,
il valore è `null` e `error` spiega il motivo.

La metrica notebook `energy` è `-cut_value`, cioè l'energia matematica
dell'obiettivo Max-Cut; **non** misura il consumo elettrico del computer o
dispositivo. Il notebook distingue ora `physical_energy_joules` e include la
media delle misure disponibili nell'analisi per grafo, algoritmo e backend.

La misura fisica CPU legge i contatori package RAPL da Linux powercap. La misura
GPU legge il contatore energetico totale NVML della GPU NVIDIA selezionata da
`backends[].params.device_index` (default `0`); installare il supporto con
`pip install -e ".[energy]"`. Entrambi misurano il package/dispositivo intero
durante il run, quindi includono attività concorrente e non attribuiscono
l'energia al solo processo. Su Windows, senza contatore compatibile, GPU non
NVIDIA e backend neuromorfico/quantum senza provider specifico, il risultato
indica `unavailable` e non usa stime.
