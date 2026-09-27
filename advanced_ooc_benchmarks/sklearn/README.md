# Incremental sklearn K-means

`../kmeans/sklearn_batching.py` reads canonical `metadata.json` and row-major little-endian FP64
`X.f64` directly. No input conversion is required. Install `requirements.txt`
in the benchmark Python environment (in Devcon, follow `devcon-env info` and
`devcon-env python ensure` first).

Training uses native `MiniBatchKMeans.partial_fit`, sequential bounded batches,
the first `k` rows as initial centers, one initialization and no random centroid
reassignment. `--passes 10` means ten **complete data passes**, not ten calls or
sklearn's default early-stopped `fit` procedure. Batch size defaults to 8,192 rows.
Centers remain FP64. Final one-based labels and final inertia are computed over
all rows in one additional input pass; labels are written to a disk-backed NPY
array. Progress is saved after every full training pass, including on a timeout.

This is a task-level comparison with Lloyd K-means: both do approximately
O(passes * rows * clusters * cols) distance work, but MiniBatchKMeans changes
centroids after every batch and does not have identical update/convergence
semantics. Compare final inertia along with runtime. Per-pass training progress
is not a completed end-to-end run (it excludes final prediction/output).

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python ../kmeans/sklearn_batching.py /path/to/dataset \
  --clusters 32 --passes 10 --output /path/to/results/sklearn.json
```

For the delegated Devcon cgroup environment, `test_memory_sweep.py` uses the
direct-cgroup helper from `../vaex/` (no Vaex dependency). It verifies enforcement
by intentionally OOM-killing a 256 MiB allocation under a 64 MiB limit, uses cold
inputs with checked per-file cache dropping, disables swap, and records elapsed
time, CPU seconds, peak cgroup memory, physical I/O and periodic telemetry.
Defaults: 16/8/4 GiB, four OpenMP threads with single-threaded BLAS, 60 seconds
per execution; tighter-memory runs are skipped after an unsuccessful pilot.
It modifies only newly created child cgroups, not ancestor limits/controllers.
The normal suite runner should be used on systemd benchmark hosts.
`--iterations` is an alias for `--passes` for runner integration; it still means
full-data passes with per-batch updates, not full Lloyd iterations. The main
benchmark plan is unchanged.

```bash
python test_memory_sweep.py --data /path/to/bench-data/dense-d32 \
  --results /path/to/bench-results-sklearn --threads 4 --timeout 60
python -m unittest discover -s . -v
```

References: [sklearn paper](https://jmlr.org/papers/v12/pedregosa11a.html),
[MiniBatchKMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).
