# Findings

Notes from adding the `random_features/` and `knn/` workloads, and from reviewing the plan's
existing cases. Two independent groups: defects in the SystemDS OOC backend, which block or
silently corrupt scripts, and observations about the benchmark design itself.

## Environment

| | |
|---|---|
| SystemDS | `9b3b7cc6aaea` (2026-08-28, *[OOC][Review] Add CorrelatedScanOOCPrimitive*), with local modifications to `BinaryOOCInstruction`, `OOCStreamable`, `SubscribableTaskQueue` |
| JVM | OpenJDK 21.0.12 (Temurin), `--add-modules=jdk.incubator.vector` |
| Blocksize | 500 in every reproduction below; the thresholds scale with it |
| Method | Each script run twice against identical inputs, once with `-exec singlenode` and once with `-exec singlenode -ooc`. CP is the reference. |

Reproductions that read an input use a 2000-by-64 dense matrix converted at blocksize 500:

```bash
python3 prepare_dense_dataset.py --out /tmp/oocbug --rows 2000 --cols 64 \
  --classes 2 --sparsity 1.0 --distribution normal --seed 7 --chunk-mib 8
SYSTEMDS_ROOT=$PREP_ROOT python3 prepare_dense_systemds.py /tmp/oocbug \
  --rows 2000 --cols 64 --blocksize 500 --chunk-mib 8 --java java \
  --systemds-jar "$SYSTEMDS_JAR" --config "$PREP_ROOT/conf/SystemDS-config.xml" \
  --java-heap 2g --java-tmp /tmp/oocbug/tmp --replace-invalid \
  --x-output /tmp/oocbug/systemds/X --binary-y-output /tmp/oocbug/systemds/by \
  --nn-y-output /tmp/oocbug/systemds/ny
```

and are then run as

```bash
java -Xmx2g --add-modules=jdk.incubator.vector \
  -cp "$SYSTEMDS_JAR:$(dirname "$SYSTEMDS_JAR")/lib/*" org.apache.sysds.api.DMLScript \
  -f repro.dml -exec singlenode -config conf.xml [-ooc] -args /tmp/oocbug/systemds/X
```

## OOC defects

Ordered by severity. The first is the dangerous one: it produces a wrong number rather than an
error, and only a second implementation to compare against reveals it.

### 1. A reshaped sequence longer than one block silently becomes zeros

`matrix(seq(a, b), rows=r, cols=c)` where the sequence spans more than one block yields a matrix
that later `+` and `/` operations treat as empty. No error, no warning. `sum` and `*` on the *same*
matrix stay correct, which is what makes it hard to spot: a checksum on the matrix looks right
while everything computed from it is wrong.

Needs no input at all:

```dml
print("small=" + sum(matrix(seq(1, 400), rows=20, cols=20) / 2));
print("large=" + sum(matrix(seq(1, 1024), rows=16, cols=64) / 2));
```

| | CP | OOC |
|---|---|---|
| `small` (400 elements, one block) | 40100.0 | 40100.0 |
| `large` (1024 elements, three blocks) | 262400.0 | **0.0** |

400 and 100 elements are correct at blocksize 500; 600 and 1024 are not. The boundary is the
blocksize, so raising the blocksize moves the failure rather than removing it.

How it surfaced: the random Fourier projection was built this way. The workload ran to completion
and reported `model_norm=0.20031503108013218` under OOC against `12.514075858770001` under CP and
`12.514075858769942` under NumPy. Nothing failed; the model was simply wrong.

**Severity: high.** Silent wrong results in a documented, natural construction.

### 2. Chained arithmetic compiles to an n-ary instruction OOC cannot emit

Three or more operands joined by `+` or `*` are folded into `n+` / `n*`, and the OOC backend has no
instruction for either:

```
DMLRuntimeException -- Only n-ary cbind, rbind, nmin, and nmax are supported: n+
```

```dml
X = read($1);
a = $2; b = $3;          # runtime scalars, so the sum is not constant-folded
Y = X + a + b;
print("sum=" + sum(Y));
```

CP prints `1023949.0347823077`; OOC fails to generate the instruction. `n*` behaves the same way,
reached through `sqrt(...) * cos(...) * scale`.

The rewriter also *creates* these chains from expressions that do not look like sums:
`column %*% ones` is replaced by a replicating multiply, which then fuses with an adjacent scalar
factor. So the constraint is not "write at most two operands per line" but "no chain of additions
or products may exist anywhere in the DAG", which is considerably harder to satisfy by inspection.

**Severity: medium.** Loud, but it constrains how any DML in this suite may be written. `cbind` and
`rbind` are supported, which is worth knowing: the feature map is built as `cbind(cos(P), sin(P))`
partly for this reason.

### 3. `outer` with a one-row left operand is rejected

A one-row `outer` degenerates to a `1x1` against `1xD` binary op, which OOC refuses:

```
NotImplementedException -- Invalid dimensions for matrix-matrix binary op: 1x1 <=> 1x64
```

```dml
X = read($1);
half = as.integer($2);
keys = outer((seq(1, 1) - 1) * half, t(seq(1, half) - 1), "+") + 99;
W = outer((seq(1, ncol(X)) - 1) * half, t(seq(1, half) - 1), "+") + 7;
print("sum=" + sum(cos(X %*% W + keys)));
```

Fails at `half=64` and at `half=1200`; the same script runs in CP. Note that the multi-row `outer`
in the same script is fine — only the degenerate one-row case fails. It also only fails when the
subgraph is OOC-assigned, i.e. when its result feeds the streamed dataflow; the same `outer` on its
own computes correctly.

**Severity: medium.** Loud, and blocks the natural way to build a broadcast row vector.

### 4. Row-slicing a matrix that spans several blocks miscounts blocks

```
DMLRuntimeException -- OOCStream block count mismatch: expected 3 but saw 1 (2x1200)
```

Same script as above with the phase built as a two-row block and sliced:

```dml
keys = outer((seq(1, 2) - 1) * half, t(seq(1, half) - 1), "+") + 99;
phase = keys[1, ];
```

| `half` | shape sliced | CP | OOC |
|---|---|---|---|
| 64 | 2x64, one block | 180.92628513783436 | 180.92628513783433 |
| 1200 | 2x1200, three blocks | -1237.8736359259437 | **block count mismatch** |

**Severity: medium.** Loud, and the threshold is again the blocksize, so a script can pass at one
blocksize and fail at another.

### What this adds up to

Small dense matrices generated inside a script are not reliable under OOC. Every one of the four
defects was hit while building a 128-by-2048 projection that has nothing to do with the streamed
data — it is generated, not read, and it is four megabytes. Three of the four are triggered by the
matrix crossing a block boundary, and one produces a silently wrong answer.

The practical consequence for this suite: `random_features/prepare_projection.py` writes the
projection to disk and all four arms read it. That is better experimental design regardless — the
projection is setup, not workload, and no engine should be measured on its random number generator
— but the reason it is not optional is this list.

The consequence for the engine: OOC assignment appears to be applied to operations that have no
business being streamed. A four-megabyte constant matrix in a script whose input is 32 GB should be
computed in CP and broadcast. Fixing the assignment heuristic would remove all four symptoms at
once, independently of fixing the individual instructions.

### Suggested reporting order

1. The silent-zero reshape (#1), because correctness defects that produce numbers outrank
   everything else.
2. The OOC assignment heuristic, as the common cause of #1, #3 and #4.
3. `n+` / `n*` instruction support (#2), which is a straightforward gap.

## Benchmark design findings

### The suite measured one regime

Before this session every dense workload had the same shape: stream a tall-skinny `X`, multiply by
an operand of at most 128 KB, reduce. Five of them (kmeans, gnmf, lmcg, l2svm, multilogreg) are that
pattern with different arithmetic in the middle. Out-of-core execution matters in three distinct
regimes, and only the first was covered:

| | input vs memory | intermediate vs memory | covered by |
|---|---|---|---|
| R1 input-bound | larger | smaller | every dense run in the plan |
| R2 intermediate-bound | **smaller** | larger | `random_features/`, `knn/` (added) |
| R3 both | larger | larger | ALS would qualify; currently disabled |

R2 is the more interesting argument: the baseline does not fail for want of RAM, it fails because
the library cannot express the computation without the user rewriting it into chunks by hand. It is
also nearly free on disk, since the large object is never stored — the two new datasets add under
3 GB against 96 GB for one dense instance.

The two new workloads deliberately bracket the same intermediate sizes (4 to 32 GB) from opposite
ends of arithmetic intensity: the Gram of the feature map is `D/4` flops per byte (128 to 512 here,
compute-dense), the distance matrix is `d/4` (128, traffic-dominated).

### Data has no structure, which limits what can be claimed

`prepare_dense_dataset.py` draws `X` from iid N(0,1) and derives the labels from a random linear
weight vector. The labels therefore carry real signal, and the supervised workloads — multilogreg,
l2svm, lmcg, and the two new ones — are learning something. `X` itself has none, so:

- k-means at k=16 on 1000-dimensional isotropic noise has no clusters to find, and at that
  dimensionality distance concentration makes assignments near-arbitrary. The comment on
  `iterations: 10` claims Lloyd's runs to convergence; on noise it does not.
- PCA takes 16 of 1000 components from a Marchenko-Pastur spectrum: about 1.6% of the variance,
  no signal to recover.
- GNMF factorizes uniform-nonnegative noise, which has no low-rank structure.

Timings remain valid — the FLOPs and the I/O are real. Reported *results* are not interpretable,
and a reviewer will say so. Adding optional cluster structure and a low-rank-plus-noise mode to the
generator costs nothing at run time and would additionally make cross-engine agreement a
meaningful correctness claim rather than agreement on noise. kNN already benefits from the label
signal: the small-scale check classifies at 83%, not 50%.

### Rank 16 everywhere makes five workloads nearly the same experiment

`clusters: 16`, `rank: 16` (SVD and GNMF), `components: 16` (PCA) all keep the model state under
128 KB, which puts those workloads at roughly 4 flops per byte of `X` — streaming passes that
differ mainly in how many times they read. PCA is the exception at about 250 flops per byte, since
`t(X)%*%X` is `O(n*d^2)`. One compute-bound point and five read-bound ones is less coverage than
the workload names suggest.

### Timeouts imply a sustained read rate

Counting `X` passes in each implementation against each `timeout_seconds`, on the 32 GB dataset:

| run | passes | bytes read | timeout | required |
|---|---|---|---|---|
| gnmf | ~40 | 1.28 TB | 1800 s | ~710 MB/s |
| multilogreg | ~40 | 1.28 TB | 1800 s | ~710 MB/s |
| lmcg | ~21 | 670 GB | 1200 s | ~560 MB/s |
| l2svm | ~21 | 670 GB | 1800 s | ~370 MB/s |
| kmeans | ~12 | 384 GB | 1200 s | ~320 MB/s |
| pca, randomized_svd | 1-2 | <=64 GB | 900 s | trivial |
| connected_components | ~6 | 190 GB | 5400 s | trivial |

Comfortable on NVMe. On a spinning disk the top three time out as a matter of arithmetic, and the
result would read as "OOC lost" when it is the storage. Worth confirming the device before a long
campaign.

### Sparse workload: prefer PageRank over connected components

PageRank runs on real Twitter-2010 data that is already prepared, has a realistic degree
distribution, and is the canonical graph benchmark. The connected-components dataset needs 1.47%
density (average degree ~6,250) because SystemDS budgets every OOC tile as dense — an engine
limitation dictating the shape of the data, which is not a footnote one wants in an evaluation
section. The two also measure much the same thing, both being sparse matrix-vector iterations. A
better second sparse workload is ALS: sparse input against dense factors, so large dense
intermediates, and a different access pattern.

### A shape sweep costs far less disk than it appears

Varying the aspect ratio at constant bytes is the cheapest way to broaden coverage, and the obvious
cost estimate is wrong: a row-major 4,000,000-by-1,000 file and a 40,000-by-100,000 file are the
*same bytes*. Only `metadata.json` and the `.mtd` differ, so one raw `X.f64` can back every shape.

| approach | four shapes at 32 GB |
|---|---|
| regenerate each shape | 384 GB |
| share the raw file | 256 GB |
| share the raw file and drop Zarr | 160 GB |

Dropping Zarr means Dask falls back to `load_matrix`, whose `from_delayed` + `concatenate` alias is
why `optimization.fuse.active` had to be disabled for arms that use it; sharing the raw file alone
is the safer saving. Wide data is the interesting row: at 40,000-by-100,000 the covariance is
100,000 squared, so PCA becomes intermediate-bound on its own and a correct implementation has to
switch to the `n`-by-`n` Gram formulation.
