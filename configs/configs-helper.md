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
  essere almeno 1. Il grafo è generato una sola volta e riusato nel piano.
  Il seed algoritmo è derivato da seed base, identificativo del grafo e
  ripetizione: algoritmi e backend confrontati ricevono lo stesso seed per
  lo stesso grafo e numero di ripetizione.
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
| `simulated_annealing` | CPU; GPU con CuPy/CUDA e catene parallele | `steps`, temperature, `gpu_replicas` |
| `exact` | CPU e GPU a batch fino a `max_nodes` | `max_nodes`, `gpu_batch_size` |
| `local_search` | CPU first-improvement; GPU best-improvement | `max_iterations` |
| `hopfield` | CPU; dinamica asincrona, aggiornamento casuale dei neuroni | `max_iterations` |
| `boltzmann` | CPU; campionamento Gibbs/heat-bath a temperatura fissa | `steps`, `temperature` |
| `spiking_neural_network` (SNN) | Simulatore LIF software su CPU; provider neuromorfico reale non disponibile | `time_steps`, `threshold`, `membrane_decay`, `synaptic_scale`, `refractory_steps` |
| `qubo` solvers | Non implementati | — |

La validazione accetta soltanto gli algoritmi registrati e implementati.
Il piano marca esplicitamente le combinazioni hardware non supportate; durante
`run` diventano errori persistiti, senza fallback.

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
`final_temperature` non deve superare `initial_temperature`. La variante GPU
usa più catene indipendenti; `gpu_replicas` imposta quante catene eseguire in
parallelo. CPU e GPU usano la stessa scala di temperature ma non la stessa
traiettoria.

### Exact: esempio per istanze piccole

```yaml
algorithms:
  - name: "exact"
    params:
      max_nodes: 20
```

L'algoritmo enumera le soluzioni, quindi il costo cresce esponenzialmente:
`max_nodes` è un limite di sicurezza, non una stima di tempo. CPU e GPU
supportano solo istanze entro tale limite; sulla GPU `gpu_batch_size` regola il
numero di soluzioni elaborate per batch. Non usare Exact sulle istanze GSET grandi.

### SNN CPU e provider neuromorfico

L'implementazione SNN disponibile è un simulatore software CPU con due neuroni
LIF competitivi per nodo, che codificano gli spin `-1` e `+1`. Produce una
soluzione Max-Cut verificata come gli altri algoritmi, ma non usa hardware
neuromorfico. Tempi e contatori di spike descrivono la simulazione CPU; non
misurano né predicono le prestazioni di un dispositivo neuromorfico.

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

`time_steps` è il numero massimo di passi, `threshold` la soglia di spike,
`membrane_decay` la perdita di potenziale per passo, `synaptic_scale` la scala
dell'input e `refractory_steps` il periodo refrattario. Il seed controlla stato
iniziale e ordine di aggiornamento. La simulazione CPU si può includere nel
confronto `configs/benchmark_heuristics_cpu.yaml`.

Il backend `neuromorphic` è intenzionalmente non disponibile finché non viene
registrato un provider hardware. Un'integrazione deve fornire un adapter
algoritmo per `spiking_neural_network` tramite
`register_backend_implementation(...)` e dichiarare il provider disponibile
con `register_backend(BackendCapability(...))`, prima di pianificare le run.
Un contatore energetico specifico si registra separatamente con
`register_physical_energy_provider(...)`. Il modulo del provider va caricato
esplicitamente dall'applicazione; il framework non simula automaticamente
l'hardware. Senza un contatore fisico, l'energia rimane `null` con stato
`unavailable`.

### Hopfield CPU

```yaml
algorithms:
  - name: "hopfield"
    params:
      max_iterations: 100000
```

Hopfield usa aggiornamenti asincroni con ordine dei neuroni casuale. La rete
usa accoppiamenti `-W`, dove `W` è la matrice pesi del grafo: ogni aggiornamento
che cambia uno spin aumenta strettamente il taglio; in caso di pareggio lo spin
resta invariato. `max_iterations` limita le valutazioni dei neuroni. Il risultato
registra aggiornamenti, flip, passate complete e convergenza. Questa variante è
CPU-only e usa lo stesso contratto del risultato normalizzato degli altri
algoritmi. Per un confronto diretto con Local Search è disponibile
`configs/benchmark_heuristics_cpu.yaml`.

### Boltzmann Machine CPU

```yaml
algorithms:
  - name: "boltzmann"
    params:
      steps: 50000
      temperature: 1.0
```

Boltzmann usa aggiornamenti Gibbs (heat-bath): a ogni passo sceglie un neurone
e lo ricampiona dalla distribuzione condizionale alla temperatura fissa
`temperature`. Restituisce lo stato con il miglior taglio incontrato durante
la catena. Si distingue da Simulated Annealing perché usa probabilità
condizionali e non proposte Metropolis con raffreddamento. `steps` conta gli
aggiornamenti dei neuroni; seed e parametri sono salvati in ogni run. È
attualmente CPU-only. L'esempio `configs/benchmark_heuristics_cpu.yaml`
confronta Boltzmann con Hopfield e Local Search.

È possibile elencare più algoritmi nello stesso file quando sono disponibili:
il piano eseguirà ciascuno su ogni grafo e backend selezionato.

## Backend hardware

Nella configurazione il campo si chiama `backends`:

```yaml
backends:
  - name: "cpu"
    params: {}
```

CPU è sempre disponibile. Le varianti GPU di Exact, Local Search e Simulated
Annealing richiedono CuPy e un dispositivo CUDA utilizzabile. Installare
`pip install -e ".[gpu]"` (extra basato su CUDA 12) e selezionare il dispositivo:

```yaml
backends:
  - name: "gpu"
    params:
      device_index: 0
```

`plan` controlla provider e disponibilità della scheda. Se CuPy o CUDA mancano,
la combinazione è marcata non supportata e `run` persiste un errore; non c'è
fallback CPU. Local Search GPU usa best-improvement invece del first-improvement
CPU. Simulated Annealing GPU usa più catene indipendenti (`gpu_replicas`), mentre
Exact enumera in batch (`gpu_batch_size`). Le strategie sono registrate nei
metadati per rendere leggibili i confronti. Quantum e neuromorfico richiedono
ancora provider dedicati per l'hardware reale; `quantum_simulator` è invece
un backend classico separato per QAOA.

```yaml
# GPU e provider hardware opzionali:
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

### QAOA: simulatore e provider quantistico

`qaoa` supporta sia il backend `quantum_simulator` sia IBM Quantum Runtime. Il
simulatore è CPU NumPy con vettore di stato di dimensione `2^n`, utile solo come
riferimento per grafi piccoli. L'implementazione IBM invia circuiti QAOA a
`SamplerV2` su un QPU reale; ogni esecuzione è stocastica e richiede accesso,
quote e tempo di coda. `max_nodes` è un limite software per il grafo. La scelta
del backend è esplicita: `quantum` non fa fallback al simulatore.

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

I risultati identificano `provider: numpy_statevector_simulator`,
`device: CPU state-vector simulation` e `quantum_hardware_used: false`.
Il contatore RAPL, se disponibile, misura il package CPU usato dal simulatore,
non hardware quantistico.

Per IBM installare `pip install -e ".[quantum]"`. L'autenticazione usa prima
`IBM_QUANTUM_TOKEN`; se non impostata, legge `apikey` dal file locale ignorato
da Git `src/apikey.json`. In alternativa si può indicare un file credenziali
esterno con `IBM_QUANTUM_TOKEN_FILE`. Non mettere token in YAML. Il canale di
default è `ibm_quantum_platform`; si può indicare un backend preciso oppure
lasciare che il provider selezioni il meno occupato con il numero di qubit
necessario:

```yaml
backends:
  - name: "quantum"
    params:
      channel: "ibm_quantum_platform"
      # backend_name: "ibm_backend_name"  # opzionale
      # instance: "instance-id"            # opzionale
```

Il piano controlla SDK e token locali. Il collegamento, l'autorizzazione e la
disponibilità del QPU vengono verificati al `run`; errori di autenticazione o
di servizio diventano run fallite con diagnosi, senza sostituire il simulatore.
I metadati riportano backend IBM, identificativi dei job, depth e shots; non
includono token. I contatori energetici QPU non sono disponibili, quindi
`physical_energy.joules` resta nullo.

`configs/benchmark_qaoa_simulator.yaml` è l'esempio QAOA simulato e
`configs/benchmark_qaoa_ibm.yaml` è l'esempio su QPU IBM.
Per confrontare gli stessi grafi con euristiche CPU usare
`configs/benchmark_qaoa_cpu.yaml`; sono configurazioni separate perché il
framework pianifica il prodotto cartesiano tra algoritmi e backend e non deve
associare QAOA al backend CPU o le euristiche al simulatore quantistico.

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

### Matrice ampia (frammento illustrativo)

Il seguente frammento mostra un confronto misto, ma non è una configurazione
eseguibile completa: Hopfield e SNN supportano solo CPU, mentre le altre tre
implementazioni supportano CPU e GPU quando CuPy/CUDA sono disponibili. Le
combinazioni senza provider vengono segnalate dal piano. Il backend neuromorfico
richiede un adapter dedicato come descritto sopra.

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

Per un confronto Hopfield/Local Search avviabile subito, usare
`configs/benchmark_heuristics_cpu.yaml`; per gli algoritmi con varianti CPU/GPU
è disponibile `configs/benchmark_cpu_gpu.yaml`.

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
