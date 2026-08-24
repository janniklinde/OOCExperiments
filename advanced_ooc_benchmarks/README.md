# Standalone advanced OOC benchmarks

This directory is the self-contained cgroup-v2 benchmark suite for the advanced dense OOC
workloads. It reuses the canonical raw dataset at `${plan.root}/bench-data/dense`. When a requested
SystemDS blocksize-qualified native representation is absent, it can prepare that representation
from the raw memmaps without modifying the raw inputs.

## Layout

- `benchmark-plan.yaml`: four currently enabled 3 GiB-heap / 4 GiB-cgroup comparisons with one
  repetition and independent SystemDS OOC and local-Spark blocksize sweeps. PageRank is retained
  but currently disabled and has a workload-specific 180-second timeout when enabled.
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
local[<threads>]`. Implementations without a blocksize association run exactly once in a separate
case such as `multilogreg_3g-baseline`. NumPy/Dask therefore continue to use the same unblocked
row-major memmaps without being repeated or labelled with a SystemDS blocksize. Before a SystemDS
case starts, the runner checks
`systemds/X-bs<blocksize>` and its matching labels and automatically creates only the missing or
incompatible representation. If OOC and Spark select the same blocksize, they share that valid
native representation. Native preparation is outside the timed cgroup scope and is logged in
`<dataset-dir>/prepare-blocksize-<blocksize>.log`. Existing valid outputs are retained, while an
incompatible output is quarantined with an `.invalid-<timestamp>` suffix. Transfer chunks remain
after a failed preparation so a retry can resume, and are removed after the final matrix validates.

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
it; merely listing `dense_2m` does not allocate data. To benchmark both sizes, create runs with
distinct IDs and set their `dataset` fields to `dense_1m` and `dense_2m`, respectively. Missing raw
files are generated first, followed on demand by every native SystemDS blocksize selected by those
runs. Incompatible raw data is never silently replaced.

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
default; PageRank overrides the timeout to 180 seconds. Local Spark uses the configured thread count
and runs its driver and executors inside the same timed cgroup. The extra cgroup GiB accommodates
JVM and Spark overhead while Linux charges and reclaims mapped input pages. Results are written
under `${plan.root}/bench-results/<run-case-id>/`.
`results.csv` reports `wall_seconds` from GNU `time`, enclosing process startup, mapping, execution,
and output. For Spark this includes `spark-submit`, JVM startup, and Spark-context initialization.
This is the primary comparison metric. `algorithm_seconds` includes workload input
setup, computation, checksums, and materialized numeric outputs, but remains a secondary metric
because framework startup boundaries differ.
