# Standalone advanced OOC benchmarks

This directory is the self-contained cgroup-v2 benchmark suite for the advanced dense OOC
workloads. It reuses the canonical raw dataset at `${plan.root}/bench-data/dense`. When a requested
SystemDS blocksize-qualified native representation is absent, it can prepare that representation
from the raw memmaps without modifying the raw inputs.

## Layout

- `benchmark-plan.yaml`: four currently enabled 3 GiB-heap / 4 GiB-cgroup comparisons with one
  repetition and independent SystemDS OOC and local-Spark blocksize sweeps. KMeans, PCA, LMCG,
  and PageRank are supported but initially disabled pending one-at-a-time host smoke runs.
- `run_cgroup_baselines.sh`, `benchmark_plan.py`, `drop_caches.py`: all runner support required
  by the plan.
- `prepare_numpy.py`: generates arbitrary-shape synthetic row-major FP64 matrices with bounded RAM
  and configurable Bernoulli sparsity.
- `prepare_dense_dataset.py`: generates the complete benchmark bundle (`X.f64`, `binary_y.f64`,
  `nn_y.f64`, and provenance metadata) from a dataset declaration.
- `convert_fp64_systemds.py`: reusable single-matrix bounded-transfer and native OOC assembly CLI.
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
- `als/` and `xgboost/` retain their DML/Python pairs for later work, but are not active in the
  plan: ALS has no forced-overflow sparse input yet, and SystemDS OOC XGBoost lacks `QSort`.
- `pagerank/` uses a deterministic, degree-irregular graph with globally dispersed edges and
  400,000-by-400,000 tiles—the same tile geometry as the Twitter PageRank experiment. Its mean
  degree of 12.5 yields about 200k nonzeros, or 3.2 MB of ultra-sparse serialized data, per tile.
  It is prepared in a separate `pagerank-b400k` dataset directory, leaving earlier PageRank
  datasets untouched.

## Comparability contract

The suite aligns major logical operations and output materialization while deliberately allowing each runtime to choose its
own physical plan. The SystemDS entrypoints import the vendored DML implementation rather than
implicitly resolving a builtin from the installed SystemDS source tree. NumPy MultiLogReg and
randomized SVD express their linear algebra over the
complete `np.memmap`; they do not perform user-controlled row blocking. Randomized SVD forms the
tall `U`, singular values, and right factor just as the DML script does, and materializes `S` and
`V`. MultiLogReg materializes its coefficient matrix. The Dask implementation uses Dask's automatic
array chunks rather than a hand-selected row partition and performs the same factor construction.

The MLP retains its 4,096-row mini-batch loop because `ffTrain` itself uses mini-batch SGD. It
matches the two affine layers, He initialization, ReLU, inverted dropout with keep probability
0.35, sigmoid/log-loss gradient, Nesterov update, momentum schedule, and learning-rate decay.
sklearn GMM receives the full memmap and materializes means and mixture weights. Its `fit_predict` path's
final label E-step matches the final E-step in DML; the baseline does not add a subsequent
`predict_proba` pass. Its EM kernels and random-number generator remain framework-specific.

PageRank uses twenty fixed power iterations, the same column-stochastic transition matrix, uniform
initial rank, and uniform dangling-mass redistribution in both arms. The SciPy baseline exposes one
whole CSR matrix over NumPy memmaps; it does not manually partition graph rows. Both arms
materialize the final rank vector in binary form.

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
local[<spark_threads>]`. The default `resources.spark_threads: 4` limits simultaneously decoded
Spark partitions independently of the host-wide CPU ceiling; this avoids exhausting the 3 GiB
driver heap with one input partition per available CPU. Implementations without a blocksize
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

The `kmeans_3g`, `pca_3g`, and `lmcg_3g` declarations are disabled by default because their large
3 GiB OOC/Spark paths have not been executed against the configured host JAR. Enable only one run
and initially reduce its sweeps to one candidate, for example:

```yaml
- id: lmcg_3g
  enabled: true
  blocksize_sweeps: {ooc: [500], spark: [1000]}
```

After that run validates, expand the sweep or enable the next workload. KMeans and LMCG use
300-second caps; covariance PCA uses a 600-second cap because its 1,024-column Gram matrix is much
more compute-intensive than the existing tall-and-skinny workloads.

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
systemctl --user status
systemd-run --user --scope -p MemoryMax=4G true
./run_cgroup_baselines.sh
```

`tools.systemds_jar` must point to an existing JAR. PageRank preparation reuses its generated
`systemds/G.ijv` import artifact when only native SystemDS conversion failed, so correct the tool
path and rerun rather than regenerating the graph.

The SystemDS templates use a 3 GiB JVM/Spark-driver heap, `MemoryMax=4G`, and a 60-second timeout by
default; PageRank overrides the timeout to 180 seconds. Local Spark defaults to four simultaneous
tasks and runs its driver and executor threads inside the same timed cgroup. Override
`resources.spark_threads` per run only after a one-case memory smoke test. The extra cgroup GiB
accommodates JVM and Spark overhead while Linux charges and reclaims mapped input pages. Results
are append-only and written under
`${plan.root}/bench-results/<YYYYMMDDTHHMMSS.microseconds+timezone>/<run-case-id>/`. The
timestamped invocation directory contains its benchmark-plan snapshot; each case directory
contains its logs, outputs, `results.csv`, and resolved context.
`results.csv` reports `wall_seconds` from GNU `time`, enclosing process startup, mapping, execution,
and output. For Spark this includes `spark-submit`, JVM startup, and Spark-context initialization.
This is the primary comparison metric. `algorithm_seconds` includes workload input
setup, computation, checksums, and materialized numeric outputs, but remains a secondary metric
because framework startup boundaries differ.
