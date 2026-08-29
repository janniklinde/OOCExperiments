# Standalone advanced OOC benchmarks

This directory is the self-contained cgroup-v2 benchmark suite for the advanced dense OOC
workloads. It reuses the canonical raw dataset at `${plan.root}/bench-data/dense`. When a requested
SystemDS blocksize-qualified native representation is absent, it can prepare that representation
from the raw memmaps without modifying the raw inputs.

## Layout

- `benchmark-plan.yaml`: aligned 4/8/16 GB synthetic inputs crossed with 16/8/4 GB memory profiles
  for MultiLogReg, randomized SVD, GNMF, KMeans, PCA, LMCG, L2-SVM, and connected components.
- `benchmark-plan2.yaml`: the previous single-configuration plan preserved unchanged.
- `run_cgroup_baselines.sh`, `benchmark_plan.py`, `drop_caches.py`: all runner support required
  by the plan.
- `setup.sh`: one-shot host preparation for the interpreter the plan names as `tools.python`.
- `docker/`: a systemd-based container that runs `run_cgroup_baselines.sh` with no host setup
  beyond a mounted dataset root and a prebuilt `SystemDS.jar`. See `docker/README.md`;
  `cd docker && cp .env.example .env && ./bench.sh build && ./bench.sh run`.
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
  whole-memmap NumPy and Zarr-backed Dask baselines.
- `prepare_zarr.py`: converts a canonical raw FP64 matrix into the uncompressed Zarr store the
  Dask arm reads, in bounded transfer bands. This is the Dask counterpart to the native SystemDS
  blocksize conversion and, like it, runs outside the timed cgroup.
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
- `connected_components/` computes connected components by max-label propagation over a sparse
  symmetric graph. `prepare.py` generates it with a planted component structure and exact ground
  truth, as CSR over raw files; SystemDS, SciPy, and Dask all read that one representation. The
  kernel is a broadcast multiply and a row-max rather than a matmul, so it exercises `ooc_uarmax`
  and `ooc_max` on sparse blocks, which no other workload here reaches.
- `random_features/` fits ridge regression on a random Fourier feature map (Rahimi & Recht,
  NIPS 2007) in the paired `[cos, sin]` form. The input is 1.02 GB and the map is 4 to 33 GB, so
  what does not fit is the intermediate rather than the data. `prepare_projection.py` writes the
  projection once per case, outside the timed scope, and all four arms read that one file.
- `knn/` classifies a fixed query block against a growing reference block by brute force. The
  query-by-reference distance matrix is the same 4 to 32 GB as the feature map above but costs
  `d/4` flops per byte to build instead of `D/4`, so the pair brackets the compute-dense and
  traffic-dominated ends of the same intermediate size.
- `prepare_sparse_systemds.py`: the sparse counterpart to `prepare_dense_systemds.py`. It reuses
  that module's staging manifest, validation, and native OOC `rbind` assembly, and differs only in
  handing each row band to the Python binding as a `scipy.sparse.csr_matrix`. Note that it must
  construct `SystemDSContext(data_transfer_mode=0)`: the default pipe transfer has no sparse
  implementation and silently hands the JVM a null `MatrixBlock`.

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

Connected components runs the vendored `m_components` label propagation in all three arms:
`u = max(rowMaxs(G * t(c)), c)` until no label changes, preceded by the same `rowSums`/`colSums`
symmetry guard. The label vector is one value per vertex, so every arm holds it densely in memory
and only the graph is out of core. The iteration count is data-determined rather than fixed, and
because all arms stop on the identical `diff == 0` test they perform the same number of passes.
Each arm reports its component count and label sum, and both are checked against the exact ground
truth the generator records, so a silently wrong arm fails instead of returning a plausible number.

The graph is sparse and is stored once, as CSR over raw files (`row_ptr.i64`, `col_idx.i32`,
`values.f64`), the representation `pagerank/` already uses. All three arms read that same
representation; only SystemDS converts, into its own blocksize-qualified sparse blocks, because it
has no reader for foreign CSR. There is deliberately no Zarr variant here: Zarr stores dense arrays,
and the point of this dataset is that the graph never exists densely. Dask's row bands are already
the natural chunking of CSR, so its tuning knob is the runtime `band_rows` parameter rather than a
prepared artifact. The 12 bytes per stored edge match what a SystemDS `SparseBlockCSR` costs (a
4-byte column index plus an 8-byte value), so the arms read comparable volume. The value array is
all ones and carries no information; it is written and read anyway, because SystemDS has no
unweighted matrix type and must read it, and synthesizing it in the baselines would cut their input
by two thirds for free. This is the opposite call from the RandomForest arm above, and for the
opposite reason: there, matching the byte volume would have meant discarding sklearn's compact
representation; here, not matching it would hand the baselines an advantage that has nothing to do
with scheduling.

Two structural properties of the graph are chosen rather than incidental. Within a component the
vertices sit on a ring and each is joined to `r ± o (mod m)` for a fixed offset set, which makes the
edge relation symmetric by construction: generation never sorts, which at billions of edges would
mean an external sort. The offset set is a short band of consecutive values plus a ladder of powers
of two. A band alone gives diameter `m/(2*band)` -- hundreds of thousands of passes. The ladder
alone gives diameter about `log2(m)`, since any offset is reachable through its binary
representation. Together, `band` tunes degree (and so dataset size) while the ladder tunes depth
(and so the pass count), and the two are independent -- which a dense adjacency cannot offer, since
there density and diameter are the same knob. Component membership and within-component rank are
both scattered by one seeded permutation; without it the ring would sit in a band around the
diagonal and the nonzeros per row chunk would be wildly uneven.

The graph's density, 1.47%, is set by SystemDS rather than chosen freely, and this is the one
number to understand before changing anything here. The OOC engine budgets every tile with
`OOCUtils.estimateOutputTileBytes`, which calls `MatrixBlock.estimateSizeDenseInMemory(blen, blen)`
-- a *dense* estimate, whatever the data's actual sparsity -- and reserves five of them against a
limit of `min(java_heap / 9, 1 GiB)`. Two consequences follow. The blocksize cannot exceed 5,181 at
any heap, and cannot exceed 2,991 under the mem4 profile's 3g heap, which is the binding case since
one blocksize serves all three profiles. And because a block spans at most about 2,991 by 2,991
cells, a block can only carry a megabyte of stored edges if the graph is denser than roughly 1%.
At blocksize 2,500 and density 1.47% each block holds 1.05 MiB, measured. The alternative -- a
web-graph-like density of 1e-5 -- puts 2.3 nonzeros in each of 1.2 billion blocks at blocksize 500,
where per-block overhead dominates completely: the same graph costs 16.13 bytes per stored edge at
304 nonzeros per block and 12.11 at 92,555. So this workload is sparse in representation and in
every arm's execution, but it is not sparse in the way a citation or web graph is, and a run whose
`java_heap` drops below 3g will need the blocksize, and hence the dataset, resized again.

The SciPy arm is present because SciPy supports the CSR layout, but it has no out-of-core execution.
As in every other NumPy arm here the update is expressed over the whole input with no
user-controlled row-block loop, which for this kernel means the gather `labels[columns]`
materializes one float per stored edge before the segment max reduces it. Nothing in the algebra
requires that temporary -- streaming it is precisely what the out-of-core arms do instead -- so the
arm is expected to complete on the smaller instances and to exhaust memory at `cc_d32`. It
deliberately does not build a `scipy.sparse.csr_matrix`: above two billion stored edges the row
pointer needs `int64`, SciPy unifies both index arrays to a single dtype, and the `int32` column
indices would be upcast, doubling the largest array in the dataset in RAM before any work starts.

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

### Intermediate-bound workloads

`random_features/` and `knn/` differ from every other workload here in what exceeds the memory
budget. Their inputs are 1.02 GB and 819 MB, so the data fits at all three profiles; the feature
map and the distance matrix do not. Neither object is ever stored on disk, which is why the pair
adds under 3 GB to the dataset root while covering intermediates from 4 to 33 GB.

Both are expressed so that nothing forces the large object to exist as a whole. `t(Z) %*% Z` and
`t(Z) %*% y` are row-separable accumulations over the feature map, and the kNN selection loop
touches the distance matrix through `rowIndexMin` and a `table` with one nonzero per query, so the
gather and the mask are each a single pass with no dense companion matrix. What each engine then
does with that freedom is the measurement: SystemDS streams, NumPy materializes, and Dask
recomputes -- the distance matrix is too large to persist under any of these budgets, so the Dask
arm rebuilds it once per neighbour and performs `k` products where the other arms perform one.
That is Dask's real behaviour for an intermediate that exceeds memory, not a handicap the script
imposes, and the arm says so in its docstring.

The kNN distance omits the query norms in all three arms. They are constant along a row, so they
shift every candidate for a query equally and cannot change which reference is nearest; the matrix
is therefore ordered like the true distance without being it, which is all selection needs.

The random Fourier projection is prepared as a file rather than generated per arm. It does not
depend on the data, it is identical for every engine, and generating it three times would compare
three random number generators instead of one feature map. `prepare_projection.py` writes it in two
encodings of the same array -- `.f64` for the NumPy and Dask arms, and `.csv` with an `.mtd`
sidecar for DML, which has no reader for raw row-major files -- at `%.17g`, so both hold the same
doubles.

#### OOC limitations these workloads ran into

Writing the feature map first surfaced four OOC gaps, all reproduced against `-exec singlenode` with
and without `-ooc` on the same inputs. They are recorded here because they constrain how any new
DML in this suite may be written, and because two of them are silent.

- Chained `+` or `*` over three or more operands compiles to an n-ary instruction that the OOC
  backend cannot generate: *"Only n-ary cbind, rbind, nmin, and nmax are supported"*. Every
  addition and product in a streamed dataflow has to stay binary. `cbind` and `rbind` are fine,
  which is one reason the map is built as `cbind(cos(P), sin(P))`.
- `matrix(seq(a, b), rows=r, cols=c)` where the sequence is longer than one block **silently
  reshapes to zeros** under OOC. `sum` and `*` on the result stay correct while `+` and `/` behave
  as if the matrix were empty, so the failure surfaces as a wrong number rather than an error.
  Reproduced at blocksize 500 with 600 and 1024 elements; 100 and 400 are correct.
- `outer(a, b, "+")` with a one-row left operand degenerates to a `1x1` against `1xD` binary op,
  which OOC rejects with *"Invalid dimensions for matrix-matrix binary op"*.
- Slicing a row out of a matrix that spans several blocks fails with *"OOCStream block count
  mismatch: expected 2 but saw 1"*.

The net effect is that small dense matrices generated inside a script are unreliable under OOC,
which is the practical reason the projection is read from disk instead. With that one change both
workloads agree across NumPy, Dask, SystemDS CP, and SystemDS OOC to the last few ulps.

### Dask input preparation

SystemDS reads a prepared native representation whose blocksize the plan sweeps; giving Dask only
the raw file would tune one runtime's physical layout and not the other's. The Dask arm therefore
reads its own prepared artifact, an uncompressed Zarr store built by `prepare_zarr.py` from the
same canonical `X.f64`, declared as the `dask_chunk` dataset variant and sized per run with
`parameters.dask_chunk`.

The store always lives at `<dataset-dir>/zarr/X.zarr`, and the Dask baselines find it there with
no argument. The chunking is recorded in the sidecar rather than in the filename, so changing
`parameters.dask_chunk` makes the preflight see a mismatch and rebuild the store in place; the
previous one is quarantined as `X.zarr.invalid-<timestamp>`. The trade-off against the
`X-bs<blocksize>` naming is deliberate: two chunk sizes cannot coexist, so sweeping `dask_chunk`
reconverts rather than reusing, which is acceptable for a value set once and outside the timed
scope. Three properties of the store are load-bearing:

- **Uncompressed.** The suite compares physical read volume, so the store must hold the same bytes
  as the raw input. Blosc/zstd on normally distributed FP64 buys almost nothing and would distort
  both `read.pdf` and the CPU chart.
- **Full-width row chunks, not square tiles.** Every kernel here is a row-local reduction
  (`t(X)%*%X`, `t(X)%*%(X%*%v)`, `t(P)%*%X`), so splitting columns forces a cross-chunk combine no
  arm's algorithm asks for. A measured sweep over 250x250, 500x500, 1000x1000, and full-width
  chunks found square tiles monotonically worse, by up to 13x at the smallest tile. Matching the
  SystemDS square blocksize would be actively unfair rather than symmetric; at 1000 columns a
  square tile also cannot exceed 8 MiB.
- **Sized by chunks per thread, and an exact divisor of the row count.** The same sweep found
  runtime tracks chunks-per-thread rather than chunk size: below roughly two chunks per thread the
  tail wave dominates and runtime degrades sharply, while anything above about six is flat. Choose
  a row chunk giving at least ~6 chunks per `resources.dask_threads`, and one that divides the row
  count exactly - Zarr writes every chunk at full size, so a trailing partial chunk is padded on
  disk and silently inflates the store (6.5% in one measured case). `prepare_zarr.py` warns and
  suggests nearby exact divisors when the chunk does not divide the shape.

The default `dask_chunk: 12500` gives 95 MiB chunks and divides 500,000 / 1,000,000 / 2,000,000 /
4,000,000 exactly. It is chosen for the 32 GB member that `dense_scaling` currently selects: 320
chunks, about 14 per thread on a 22-thread host. **If `dense_scaling` is widened back to
`dense_d4`, revisit it** - 500,000 rows at 12500 is only 40 chunks, under two per thread on the
same host, which the sweep puts in the degraded region.

Only `X` is converted. The response vectors are one column wide and are read eagerly as NumPy by
`dask_support.read_vector`: chunking them to match `X` would add one kilobyte-sized task per row
block of the large matrix to every graph, for an operand that is 32 MB beside a 32 GB input.

Because the Zarr array carries no alias layer, low-level fusion is safe and stays enabled. The
older `load_matrix` reader built the array from `from_delayed` plus `concatenate`, which puts an
alias at the top whose task specification is the *name* of another key; fusion renames that key
differently depending on the downstream graph, so reusing one such array across the fixed-iteration
submissions made the scheduler see two specifications for one stable key and warn about possible
deadlocks. `load_matrix` remains for workloads that have not been converted; a plan that falls back
to it must also set `optimization.fuse.active: False`.

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
PCA scores and GNMF `W` are streamed into an uncompressed Zarr store rather than first being
collected into the Python heap. They previously targeted an `np.lib.format.open_memmap`, which
does not work under a distributed client: the memmap is serialized to the worker, which writes
into its own copy, so the file stayed zero-filled while the returned checksum was still correct. NumPy continues to use `resources.threads`; its activation-sized MLP
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
metadata. NumPy and dense SciPy operations consume it directly through `np.memmap`. Dask reads a
prepared uncompressed Zarr store instead; see "Dask input preparation" below. Sparse SciPy workloads instead use canonical
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
./setup.sh
systemd-run --user --scope -p MemoryMax=4G true
./run_cgroup_baselines.sh
```

`setup.sh` installs `requirements-baselines.txt` into the interpreter configured as `tools.python`
rather than the active `python3`, creating that interpreter as a virtualenv when the path does not
exist yet. It then installs the SystemDS Python bindings from the source tree next to
`tools.systemds_jar` (dataset preparation needs `import systemds`; pass `--skip-systemds-python` to
manage them yourself), imports every module any implementation declares — including the ones behind
currently disabled runs, so enabling a run later does not send you back — and reports on `java`,
`spark_submit`, `tools.systemds_jar`, user systemd, and the writability of `root`. External tools
are warnings rather than failures: a missing `spark_submit` only matters once a Spark
implementation is enabled. It ends by validating the plan, and re-running it is safe.

### Enable cgroup-v2 `io.stat` accounting

The runner already creates every timed scope with `IOAccounting=yes`; no YAML setting is needed.
For `io.stat` to exist inside a *user* scope, however, the host system manager must delegate the
`io` controller through `user@<uid>.service`. On this host the vendor unit delegated only `cpu`,
`memory`, and `pids`, so `IOAccounting=yes` alone could not create an `io.stat` file.

As an administrator, create `/etc/systemd/system/user@.service.d/90-io-delegation.conf`:

```ini
[Service]
Delegate=
Delegate=cpu io memory pids
```

The empty `Delegate=` resets the vendor setting before the replacement list is applied. Reload the
system manager and restart the user manager/session (a reboot is the simplest reliable choice when
running a graphical login):

```bash
systemctl daemon-reload
# reboot, or otherwise restart the affected user@<uid>.service and log in again
```

After re-login, verify that the effective delegation includes `io`:

```bash
systemctl show "user@$(id -u).service" --no-pager \
  -p Delegate -p DelegateControllers
# expected: DelegateControllers=cpu io memory pids
```

Finally, verify the exact kind of scope used by the runner rather than inspecting only the terminal
scope. This command should print `io.stat available`; an empty `io.stat` is valid for a command
that did no physical block I/O, while a missing file is not.

```bash
systemd-run --user --scope --expand-environment=no -p IOAccounting=yes \
  bash -c '
    cg=$(grep "^0::" /proc/self/cgroup | cut -d: -f3-)
    root="/sys/fs/cgroup${cg}"
    test -e "$root/io.stat" && echo "io.stat available" || exit 1
    cat "$root/io.stat"
  '
```

If this still fails, inspect the controller chain at
`/sys/fs/cgroup/user.slice/user-<uid>.slice/user@<uid>.service`: `io` must be available to the
user manager. The benchmark emits a one-time warning when its actual timed scope lacks
`io.stat`; in that state GNU `time` remains available, but `results.csv` cannot report
byte-accurate cgroup read/write counters.

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
