# K-means baselines

`systemds.dml`, `numpy.py`, and `dask_array.py` implement fixed-iteration Lloyd
K-means. `sklearn_batching.py` is a separate **MiniBatchKMeans** task-level baseline,
not an identical Lloyd implementation. It is not yet enabled in the main plan.

It streams contiguous slices of the existing row-major FP64 `X.f64` memmap into
native sklearn `partial_fit`: no data conversion, full-input copy, Python worker
pool or custom prefetch layer is needed. The OS handles paging/readahead. This is
an appropriate simple loader for an already-prepared local dense matrix. Batch
size is configurable because the same number of rows can have very different
memory footprints for different feature counts.

Initialization uses the first `k` rows, one initialization, and no random center
reassignment. Ten passes process every row ten times; updates occur after each
batch. Final centers, one-based labels and final inertia are materialized using
one additional bounded-memory pass. Compare both runtime and final inertia with
Lloyd, not only their iteration counts. Sequential batches intentionally avoid
an input-sized shuffled copy or random disk accesses; this can influence the
optimization trajectory on ordered datasets.

From the repository root, using the interpreter with sklearn installed:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python advanced_ooc_benchmarks/kmeans/sklearn_batching.py \
  /path/to/bench-data/dense-d32 --clusters 32 --passes 10 --batch-rows 8192 \
  --output /path/to/results/sklearn.json
```

`--iterations` aliases `--passes`; neither enables default sklearn early stopping.
Thread counts are controlled through the environment to avoid nested OpenMP/BLAS
oversubscription. Dependencies, the experimental memory-sweep driver and numerical
tests remain under `../sklearn/`; see its README. Output JSON includes complete-pass
progress, FP64 dtype, batch size, sklearn version, training time and final objective.

The standalone command does not enforce memory limits or cold caches; use a
cgroup runner for comparable memory-sweep measurements.
