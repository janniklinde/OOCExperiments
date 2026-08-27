# Standalone advanced OOC benchmarks

This directory is the self-contained cgroup-v2 benchmark suite for the advanced dense OOC
workloads. It reuses the canonical raw dataset at `${plan.root}/bench-data/dense`. When a requested
SystemDS blocksize-qualified native representation is absent, it can prepare that representation
from the raw memmaps without modifying the raw inputs.

## Layout

- `benchmark-plan.yaml`: aligned 4/8/16 GB synthetic inputs crossed with 16/8/4 GB memory profiles
  for MultiLogReg, randomized SVD, GNMF, KMeans, PCA, LMCG, and L2-SVM.
- `benchmark-plan2.yaml`: the previous single-configuration plan preserved unchanged.
- `run_cgroup_baselines.sh`, `benchmark_plan.py`, `drop_caches.py`: all runner support required
  by the plan.
- `prepare_numpy.py`: generates arbitrary-shape synthetic row-major FP64 matrices with bounded RAM
  and configurable Bernoulli sparsity.
- `prepare_dense_dataset.py`: generates the complete benchmark bundle (`X.f64`, `binary_y.f64`,
  `nn_y.f64`, and provenance metadata) from a dataset declaration.
- `convert_fp64_systemds.py`: reusable single-matrix bounded-transfer and native OOC assembly CLI;
  FP64 remains the default, while `--dtype` supports compact prepared inputs such as binned `uint8`.
- `prepare_dense_systemds.py`: transfers canonical raw matrices in bounded, block-aligned chunks,
  then uses native SystemDS OOC `rbind` and an explicit-blocksize write to create
  `X-bs<blocksize>`, `binary_y-bs<blocksize>`, and `nn_y-bs<blocksize>`.
- One directory per workload. `systemds.dml` is the runnable SystemDS entrypoint; where a builtin
  is used, `implementation.dml` is its vendored implementation and is imported with DML
  `source(...) as module`. `numpy.py` is the matching NumPy baseline. Randomized SVD additionally
  contains `dask_array.py`; it is not named `dask.py` because that would shadow the installed
  Python `dask` package.
- `kmeans/`, `pca/`, and `lmcg/` contain deterministic self-contained DML implementations plus
  whole-memmap NumPy and automatically chunked Dask baselines.
- `gnmf/` implements the fixed-iteration Lee-Seung multiplicative updates over a non-negative FP64
  matrix. SystemDS, NumPy, and Dask share deterministic positive initialization and materialize both
  learned factors.
- `als/` uses an auto-prepared deterministic sparse ratings matrix and a SciPy ALS-CG baseline.
  Its canonical CSR arrays are generated once, while blocksize-qualified native SystemDS inputs
  are converted on demand for each OOC/Spark blocksize candidate. Its entrypoint calls the vendored
  `m_alsCG` implementation directly, so the complete algorithm being measured is available beside
  the benchmark just like the MLP implementation sources.
- `randomforest/` uses a bounded-memory prepared 8-bin `uint8` matrix in its own
  `bench-data/randomforest` directory. SystemDS
  and sklearn both train fixed-depth, Gini-classification forests without row bootstrapping and
  materialize their learned models.
- `pagerank/` reuses the prepared Twitter-2010 graph from `experiments/real_world`, including its
  deterministic vertex permutation, normalized CSR representation, dangling bitmap, and native
  400,000-by-400,000 SystemDS tiles used by the established Twitter experiments.

## Comparability contract

The suite aligns major logical operations and output materialization while deliberately allowing each runtime to choose its
own physical plan. The SystemDS entrypoints import the vendored DML implementation rather than
implicitly resolving a builtin from the installed SystemDS source tree. NumPy MultiLogReg and
randomized SVD express their linear algebra over the
complete `np.memmap`; they do not perform user-controlled row blocking. Randomized SVD forms the
tall `U`, singular values, and right factor just as the DML script does, and materializes `S` and
`V`. MultiLogReg materializes its coefficient matrix. The Dask implementation uses Dask's automatic
array chunks rather than a hand-selected row partition and performs the same factor construction.

The MLP uses one 65,536-row batch and a configurable 9,216-neuron hidden layer. Each dense hidden
activation contains 603,979,776 FP64 values (4.5 GiB), and forward/backward training creates several
such logical tensors. The input itself is intentionally only 65,536 by 8: this keeps the arithmetic
manageable while making the neural-network activation working set, rather than merely the source
dataset, exceed the 4 GiB cgroup. The weights remain small, so this is accurately an activation-OOC
benchmark rather than an OOC model-parameter benchmark. NumPy expresses the same full batch using
disk-backed activation, mask, and gradient arrays; Dask derives chunks automatically from the
expanded activation geometry.
All implementations retain the two affine layers, He initialization, ReLU, inverted dropout with
keep probability 0.35, sigmoid/log-loss gradient, Nesterov update, momentum schedule, and
learning-rate decay. Framework-specific dropout random streams can produce different model values.
sklearn GMM receives the full memmap and materializes means and mixture weights. Its `fit_predict` path's
final label E-step matches the final E-step in DML; the baseline does not add a subsequent
`predict_proba` pass. Its EM kernels and random-number generator remain framework-specific.

PageRank uses fifteen fixed power iterations, the same column-stochastic transition matrix, uniform
initial rank, and uniform dangling-mass redistribution in both arms. The SciPy baseline exposes one
whole CSR matrix over NumPy memmaps; it does not manually partition graph rows. Both arms
materialize the final rank vector in binary form.

RandomForest consumes the identical prepared 1-through-8 feature bins in both arms. Both use Gini
classification, all training rows and columns per tree, square-root feature candidates per split,
the same tree count, depth, minimum leaf size, and minimum split size. Framework-specific random
streams and tree-layout representations may differ; both outputs contain the complete trained
forest rather than only predictions or a checksum. This is logical-work comparability, not equal
input-byte comparability: sklearn maps the one-byte bins directly, while native SystemDS binary
blocks represent matrix values as doubles. Cold-cache wall time therefore includes different input
volumes; use `algorithm_seconds` as supporting evidence or prepare an FP64 sklearn input when equal
physical read volume is required.

KMeans uses the first `k` records as deterministic initial centroids and performs the same fixed
number of Lloyd iterations in SystemDS, NumPy, and Dask. Every iteration computes the complete
record-to-centroid distance matrix and weighted centroid update; all arms then perform a final
assignment pass and materialize centroids and one-based labels. This avoids framework-specific
random initialization and convergence criteria.

PCA is centered but unscaled. All arms compute column means, covariance through the `t(X)%*%X`
identity without constructing a second input-sized centered matrix, dominant eigenvectors, and the
complete projected score matrix. Scores, components, and eigenvalues are materialized. Eigenvector
signs may differ while representing the same components.

LMCG solves the same zero-intercept, L2-regularized normal equations using conjugate gradient. It
starts from zero, uses the correlated `binary_y.f64` response, performs the declared fixed number
of `X%*%p` and `t(X)%*%v` passes when tolerance is zero, and materializes the coefficient vector.
NumPy exposes the complete memmap to its linear algebra kernels; Dask selects chunks automatically.

L2-SVM trains the same zero-intercept binary linear classifier with squared hinge loss and L2
regularization. SystemDS calls the vendored builtin implementation directly; NumPy and Dask mirror
its nonlinear conjugate-gradient direction and Newton line search. All arms use the correlated
`{-1,+1}` labels, zero initialization, three outer iterations, 20 inner iterations, zero stopping
tolerance, and materialize the complete coefficient vector. NumPy exposes the whole memmap and
Dask chooses chunks automatically; neither baseline introduces a user-selected row-block loop.

GNMF minimizes the Euclidean reconstruction error of `X ~= W %*% H` with the standard Lee-Seung
multiplicative updates. Its dedicated 1,000,000-by-384 input is uniformly distributed on `[0,1)` and
occupies 3.072 GB as raw FP64. Rank, iteration count, epsilon, seed, update order, initialization,
and full `W`/`H` output materialization match across SystemDS, NumPy, and Dask. Each iteration's
`t(W)%*%X` and `X%*%t(H)` operations provide two canonical scans of the large input.

The dense-data preflight requires the canonical raw FP64 files, derives their exact byte sizes from
the declared shape, checks dimensions, seed, sparsity, distribution, and generator version, and
validates the corresponding SystemDS matrix metadata for each run's blocksize. Cache eviction is a
hard precondition: a run aborts instead of silently comparing a cold implementation with a warm
one.

Dense blocksize is a backend-specific run setting, not part of the logical dataset. OOC and local
Spark candidates are configured independently:

```yaml
blocksize_sweeps:
  ooc: [250, 500, 1000]
  spark: [1000, 2000]
```

A sweep expands into independently named cases such as `multilogreg_3g-ooc-bs500` and
`multilogreg_3g-spark-bs2000`, with separate result directories and CSV files. The OOC case contains
`systemds-ooc`; the Spark case contains `systemds-spark`, launched using `spark-submit --master
local[<spark_threads>]`. The scaling plan gives SystemDS, NumPy, Dask, and Spark the same host-wide
thread allowance through `auto`; Dask fixes BLAS to one thread per scheduled task to avoid nested
parallelism. Implementations without a blocksize
association run exactly once in a separate case such as `multilogreg_3g-baseline`. NumPy/Dask
therefore continue to use the same unblocked
row-major memmaps without being repeated or labelled with a SystemDS blocksize. Before a SystemDS
case starts, the runner checks
`systemds/X-bs<blocksize>` and its matching labels and automatically creates only the missing or
incompatible representation. If OOC and Spark select the same blocksize, they share that valid
native representation. Native preparation is outside the timed cgroup scope and is logged in
`<dataset-dir>/prepare-blocksize-<blocksize>.log`. Existing valid outputs are retained, while an
incompatible output is quarantined with an `.invalid-<timestamp>` suffix. Transfer chunks remain
after a failed preparation so a retry can resume, and are removed after the final matrix validates.
Cold-cache traversal ignores preparation symlinks to tools such as the SystemDS JAR.

The implementation template's `blocksize_sweep` field associates it with `ooc` or `spark`.
Implementations without an association, such as NumPy, sklearn, SciPy, and Dask, are grouped in the
single `-baseline` case. This case skips SystemDS native conversion and SystemDS-specific setup.
The dedicated Dask template uses one in-process threaded scheduler, limits it with
`resources.dask_threads`, targets `resources.dask_chunk_size` per automatically selected chunk,
and fixes BLAS libraries to one thread per task, avoiding nested Dask-by-BLAS parallelism. The
default 32 MiB target controls task granularity, while each memory profile scales Dask's managed
memory limit to 12, 6, or 3 GiB; neither setting is a SystemDS blocksize. Tall Dask outputs such as
PCA scores and GNMF `W` are stored directly into `.npy` memmaps rather than first being collected
into the Python heap. NumPy continues to use `resources.threads`; its activation-sized MLP
intermediates and tall PCA output are disk-backed memmaps. RandomForest separately caps concurrent
sklearn tree builds with `resources.python_jobs`, because job-level parallelism duplicates
tree-building state. Spark uses its independent `resources.spark_threads` setting.
The old scalar or list-valued `parameters.blocksize` form remains accepted for plans that want one
shared sweep across all implementations.

Dense datasets are declared once with a YAML anchor. The existing `dense_1m` is explicitly allowed
to use its legacy metadata, whose seed and sparsity cannot be independently verified. New datasets
must use `accept_legacy: false`, which makes their full generation identity mandatory. For example:

```yaml
datasets:
  dense_1m: &dense_dataset
    directory: ${plan.root}/bench-data/dense
    parameters:
      {rows: 1000000, cols: 1024, classes: 2, sparsity: 1.0,
       distribution: normal, seed: 7, generator_chunk_mib: 256,
       accept_legacy: true, systemds_chunk_mib: 128}
    # ready, artifacts, and prepare are defined here; see benchmark-plan.yaml.

  dense_2m:
    <<: *dense_dataset
    directory: ${plan.root}/bench-data/dense-2m
    parameters:
      {rows: 2000000, cols: 1024, classes: 2, sparsity: 1.0,
       distribution: normal, seed: 17, generator_chunk_mib: 256,
       accept_legacy: false, systemds_chunk_mib: 128}
```

Preparation has `policy: auto`. A declared dataset is generated only when an enabled run references
it; merely listing `dense_2m` does not allocate data. A run can select one dataset or a non-empty
list of datasets:

```yaml
- id: multilogreg_3g
  dataset: [dense_1m, dense_2m]
  blocksize_sweeps: {ooc: [500, 1000], spark: [1000]}
```

Dataset lists expand before backend/blocksize sweeps. The example therefore produces cases such as
`multilogreg_3g-dense_1m-baseline`, `multilogreg_3g-dense_1m-ooc-bs500`, and
`multilogreg_3g-dense_2m-spark-bs1000`. A scalar `dataset: dense_1m` retains the existing names
without a dataset suffix. Empty lists, duplicate dataset IDs, and unknown datasets are rejected
during plan validation. Missing raw files are generated per dataset first, followed on demand by
every native SystemDS blocksize selected for that dataset. Incompatible raw data is never silently
replaced.

Larger campaigns should use named dataset groups, resource profiles, and correlated parameter
cases instead of duplicating run declarations. The default plan includes an aligned 4/8/16 GB
`dense_scaling` group, consistent cgroup/JVM pairs, and activation-sized MLP cases:

```yaml
resource_profiles:
  mem16: {memory_max: 16G, java_heap: 12g, dask_memory_limit: 12GiB}
  mem8: {memory_max: 8G, java_heap: 6g, dask_memory_limit: 6GiB}
  mem4: {memory_max: 4G, java_heap: 3g, dask_memory_limit: 3GiB}

dataset_groups:
  dense_scaling: [dense_d4, dense_d8, dense_d16]

parameter_cases:
  mlp_activation:
    - {id: act4, hidden_size: 8192}
    - {id: act8, hidden_size: 16384}

runs:
  - id: lmcg
    dataset: {group: dense_scaling}
    resource_profiles: [mem16, mem8, mem4]
    blocksize_sweeps: {ooc: [500], spark: [1000]}
    # Remaining workload fields are unchanged.
```

Expansion order is dataset, resource profile, parameter case, backend/blocksize, and repetition.
For example, a run that also selects `parameter_cases: mlp_activation` can produce
`mlp-dense_d8-mem4-act8-ooc-bs500`. A selected resource profile overrides matching keys in the
run's `resources` mapping; unrelated run-specific settings such as its timeout remain in effect.
Parameter cases similarly override matching keys in `parameters`. Scalar datasets and runs without
these selectors retain their previous names and behavior. Dataset groups, profiles, parameter case
IDs, duplicates, and all referenced datasets are validated even when the corresponding run is
disabled.

The current plan enables MultiLogReg, randomized SVD, GNMF, KMeans, PCA, LMCG, and L2-SVM. With
`dense_scaling` and `gnmf_scaling` currently narrowed to their 16 GB members, three resource
profiles and three backend case types expand these to 63 run cases and 81 implementation executions
per repetition. GNMF uses its separate size-matched nonnegative dataset. For an initial host smoke
test, disable the other workloads and retain one dataset and profile, for example:

```yaml
- id: lmcg_scaling
  enabled: true
  dataset: dense_d4
  resource_profiles: [mem16]
  blocksize_sweeps: {ooc: [500], spark: [1000]}
```

KMeans, LMCG, and L2-SVM use 300-second caps; covariance PCA and GNMF use 600-second caps. Dataset
generation and native conversion remain on demand and outside the measured cgroups.

The canonical dense interchange format is headerless little-endian row-major FP64 plus JSON
metadata. NumPy, Dask, and dense SciPy operations can all consume it directly through `np.memmap`;
Dask wraps that memmap with `dask.array.from_array`. Sparse SciPy workloads instead use canonical
binary CSR arrays (`row_ptr`, `col_idx`, and `values`) because zeros in raw dense FP64 still occupy
eight bytes each. HDF5 or `.npy` copies are optional and are not required by this suite.

For example, this generates a 10,000,000 by 1,024 matrix at expected sparsity 0.1 while limiting
the generator's primary working arrays to approximately 256 MiB:

```bash
python prepare_numpy.py /data/X.f64 --rows 10000000 --cols 1024 \
  --sparsity 0.1 --seed 7 --chunk-mib 256
```

To convert one such file independently of the benchmark plan, provide the SystemDS configuration
that should also be used by the final OOC assembly:

```bash
SYSTEMDS_ROOT=/path/to/systemds-root python convert_fp64_systemds.py \
  /data/X.f64 /data/systemds/X-bs500 --rows 10000000 --cols 1024 \
  --blocksize 500 --chunk-mib 128 --systemds-jar /path/to/SystemDS.jar \
  --config /path/to/SystemDS-config.xml
```

## Run

Update `root` and `tools` in `benchmark-plan.yaml` for the target host. In particular,
`tools.spark_submit` must resolve to a Spark 3.x `spark-submit` executable. Then verify user-systemd
cgroups are available:

```bash
python3 -m pip install -r requirements-baselines.txt
systemctl --user status
systemd-run --user --scope -p MemoryMax=4G true
./run_cgroup_baselines.sh
```

Install the requirements with the same interpreter configured as `tools.python`; the command above
is illustrative when that interpreter is the active `python3`.

`tools.systemds_jar` must point to an existing JAR. PageRank preparation delegates to
`experiments/real_world/prepare.sh`, whose downloads are resumable and whose expensive CSR data is
reused when only a native SystemDS representation is missing.

The scaling plan pairs 12/6/3 GiB Java heaps with 16/8/4 GB cgroups. Local Spark runs its driver,
executor threads, native allocations, and local shuffle storage inside the same timed cgroup. Its
task count and blocksize are independent from the OOC settings. Each case has private SystemDS
scratch, Java temporary, and Spark-local directories,
which prevents stale temporary state from another case from entering a lazy Spark lineage. Override
`resources.spark_threads` per run only after a one-case memory smoke test. Heap headroom accommodates
JVM/native overhead while Linux charges and reclaims mapped input pages. Results
are append-only and written under
`${plan.root}/bench-results/<YYYYMMDDTHHMMSS.microseconds+timezone>/<run-case-id>/`. The
timestamped invocation directory contains its benchmark-plan snapshot and `expanded-plan.yaml`,
whose runs have no remaining group/profile/case selectors and whose implementations have their
templates resolved. Its compact `invocation-metadata.json` records the host CPU/kernel/memory and
filesystem, runner and configured tool paths, the SystemDS JAR SHA-256, and SHA-256 provenance for
every benchmark-suite source file. Each case directory contains its logs,
outputs, `results.csv`, `resolved-context.json`, and `resolved-run.json`. The latter records concrete
inputs, resource settings, setup commands, and per-repetition commands without copying the inherited
host environment.
Every timed implementation/repetition receives fresh private `systemds-tmp`, `systemds-scratch`,
`spark-local`, `java-tmp`, `dask-spill`, and `python-tmp` roots below its case directory. The runner
removes those complete roots in a `finally` block after success, failure, timeout, or cgroup OOM;
cleanup occurs after the measured wall-time boundary. Logs, metrics, and declared outputs are not
part of the cleanup set. Plans can override the list with top-level or run-level `temporary_paths`,
but the runner rejects the case directory itself and every path outside it.
`results.csv` reports `wall_seconds` from GNU `time`, enclosing process startup, mapping, execution,
and output. For Spark this includes `spark-submit`, JVM startup, and Spark-context initialization.
This is the primary comparison metric. `algorithm_seconds` includes workload input
setup, computation, validation invariants, and materialized numeric outputs, but remains a secondary metric
because framework startup boundaries differ.

Each execution also writes a compact `*.telemetry.csv` sampled from its cgroup every second. It
records current and peak charged memory; anonymous, file-cache, shared, dirty, and writeback bytes;
page faults and working-set refaults; CPU user/system time and throttling; process count; physical
read/write bytes and operations; and CPU, memory, and I/O pressure-stall totals. The final counter
values are copied into `results.csv` beside GNU time's faults and filesystem-I/O counters, while the
raw cgroup `memory.events` and per-device `io.stat` remain in `*.metrics`. These measurements cover
the complete process tree, including Spark threads and child processes. Generic Linux page-cache
hits are inferred from physical I/O and refault counters; SystemDS's exact buffer-pool cache hits,
evictions, spill volume, OOC heavy hitters, and timings remain in its `-oocStats` log.

After every comparable backend for a workload/dataset/memory/repetition group has run, the runner
compares compact numerical invariants. If all expected executions completed and agree within the
declared tolerance, it deletes their materialized numeric artifacts outside the measured interval
and retains JSON summaries plus `output-retention.json`. A failure, missing validation evidence, or
numerical divergence preserves every artifact in that comparison group. Unsupported workloads are
also preserved conservatively. The invocation-wide decisions are recorded in
`output-validation.json`.

At one sample per second a row is roughly a few hundred bytes: even 81 executions that each consume
their full 3,600-second timeout remain well below 250 MiB of telemetry. This estimate excludes
declared numeric model/output artifacts. In particular, PCA scores and GNMF factors are deliberately
materialized for comparability and can themselves consume several GiB across the complete sweep.

When Python or Dask templates are enabled, the runner checks their
declared modules in `tools.python` and reports a single plan error for missing NumPy, SciPy,
scikit-learn, joblib, or Dask dependencies. The timed payload raises its Linux OOM score relative to
the accounting wrapper, so a cgroup OOM normally leaves exit status 137, GNU-time data, and cgroup
memory events instead of an empty log and `nan` metrics. NumPy MLP activation memmaps use the
runner-owned `python-tmp` root and are therefore removed after failed or successful executions.
