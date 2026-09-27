# Separate Vaex OOC baselines

Native `vaex.ml.cluster.KMeans` and `vaex.ml.transformations.PCA` baselines.
The main `../benchmark-plan.yaml` and its environments are deliberately unchanged.
These are the first two additional framework baselines; Block LIBLINEAR, sklearn,
and the other proposed solvers are not integrated here yet.

## Installation

Use a **separate Python environment**: Vaex 4.19 requires Dask <2024.9 and pandas <3,
which conflicts with the main suite. Python 3.11 has a tested binary wheel.

```bash
python3.11 -m venv ~/.venvs/vaex-bench
~/.venvs/vaex-bench/bin/python -m pip install -r advanced_ooc_benchmarks/vaex/requirements.txt
```

In the Devcon environment used for development, the tested interpreter is
`/opt/devcon/env/vaex/bin/python` (run `devcon-env info` and
`devcon-env python ensure` before installing dependencies on another Devcon host).

## Reuse existing datasets

Input is the canonical `metadata.json` plus **little-endian, row-major FP64
`X.f64`**, just like the existing NumPy baselines. No new random data is generated.

* `--layout raw`: direct, zero-copy column views into the existing row-major
  memmap; no preparation or extra input disk space. Column scans are strided.
* `--layout columnar`: use the same values in a Fortran-order FP64 `.npy` file
  at `<dataset>/vaex/X.npy`, giving Vaex contiguous mapped columns. Preparation
  takes one source pass, needs another input-sized file, and uses bounded transfer
  bands (64 MiB by default, or one row if wider). It is lossless, not quantized.
  File identity, size, mtime and metadata are recorded; stale derived data is
  regenerated, without deleting canonical inputs. Do not modify inputs in place
  while preserving their timestamp. Run only one preparation per dataset at a time.

```bash
~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/vaex/prepare.py \
  /path/to/bench-data/dense-d32

~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/vaex/kmeans.py \
  /path/to/bench-data/dense-d32 --layout columnar --threads 4 \
  --clusters 32 --iterations 10 --output /path/to/results/kmeans.json

~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/vaex/pca.py \
  /path/to/bench-data/dense-d32 --layout columnar --threads 4 \
  --components 16 --output /path/to/results/pca.json
```

The standalone commands do not impose a memory limit or drop caches themselves.
`--chunk-rows` defaults to 65,536 and bounds native execution chunks and output
transfer chunks; concurrent tasks still need headroom proportional to feature
count and worker count. PCA retains a columns-by-columns covariance/eigenvector
working set: this is **data-OOC**, not an OOC eigensolver for arbitrarily wide data.

## Run with the existing cgroup measurement infrastructure

Generate a separate plan. Repeat `--data` to select several already-prepared
datasets. Columnar conversion is automatically performed in untimed setup, before
cold-cache handling and the measured scope. Raw layout skips it entirely.

```bash
~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/vaex/make_plan.py \
  --data /path/to/bench-data/dense-d32 \
  --results /path/to/bench-results-vaex \
  --memory 16G 8G 4G --threads 4 --clusters 32 --iterations 10 \
  --components 16 --layout columnar \
  --output /path/to/vaex-plan.yaml

~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/benchmark_plan.py \
  /path/to/vaex-plan.yaml --validate

~/.venvs/vaex-bench/bin/python advanced_ooc_benchmarks/benchmark_plan.py \
  /path/to/vaex-plan.yaml
```

This uses the existing runner's dated invocation/case directories, memory/swap
limits, timeouts, `/usr/bin/time`, and periodic cgroup telemetry including
`io.stat` when available. The normal host requirements for systemd user scopes
and cache dropping still apply. Outputs are written with native baseline names;
this separate plan **does not automatically compare against another invocation's
outputs**. Keep its artifacts for cross-framework validation. Preparation uses
file-backed output pages, so its RSS/page-cache footprint is not strictly capped
by the transfer-band size; it runs outside the measured cgroup like other staging.

## Comparability and explicit differences

* K-means starts from the first `k` rows, `n_init=1`, with exactly the requested
  Lloyd iteration count. A tiny subclass overrides only Vaex's early-stop check;
  distance computation, centroid accumulation and reduction are native Vaex.
  Final centers and one-based labels are saved. Final inertia is computed from
  the same bounded chunks used to materialize native predictions, without an
  extra input scan (fit-time inertia refers to centers before the last update).
  Ties use Vaex's first-centroid policy, whereas the
  existing suite splits ties during fitting and chooses the last label. Continuous
  synthetic inputs normally avoid this difference. Empty clusters cause an explicit
  failure rather than silently accepting NaN centers.
* PCA is centered, unscaled and unwhitened, using Vaex's native covariance PCA,
  **not IncrementalPCA**. Its population-covariance eigenvalues are multiplied
  by `n/(n-1)` to match the suite's sample covariance; eigenvectors and scores
  are unchanged. Components, eigenvalues and the entire FP64 score matrix are
  materialized. Native Vaex requires at least two requested components.
* Both reports include framework versions, thread count, layout, chunk size and
  elapsed script time. Use runner wall time as the primary end-to-end metric.
  Native Numba JIT compilation is included in a fresh K-means process; do not
  compare this silently against an in-process warm execution.
* Tests check both layouts against the existing NumPy scripts. Eigenvector signs
  are aligned for comparison; degenerate PCA eigenspaces require subspace checks.

```bash
~/.venvs/vaex-bench/bin/python -m unittest discover \
  -s advanced_ooc_benchmarks/vaex -v
```

References: [Vaex paper](https://arxiv.org/abs/1801.02638),
[official ML documentation](https://vaex.io/docs/tutorial_ml.html),
[native K-means implementation](https://github.com/vaexio/vaex/blob/master/packages/vaex-ml/vaex/ml/cluster.py),
[native PCA implementation](https://github.com/vaexio/vaex/blob/master/packages/vaex-ml/vaex/ml/transformations.py).
