# Concurrent packing: iterative projection and contraction

This experiment runs `Y = X %*% P; P = t(X) %*% Y; P = P / max(abs(P))`
three times. X is dense FP64 with two columns, P has eight columns, and the
logical blocksize is 1000. It tests the actual SystemDS cache, spill serializer,
allowances, leases, MatrixBlock kernels and completed intermediate stores.
It is a Java operator-composition experiment, not a planner-selected DML job,
full KMeans implementation, or comparison against the existing unpacked DML plan.

## Implementation boundaries

- `PackingMaterializer` owns independent producer lanes. Each lane accepts an
  owned tile and an opaque locality key, accumulates references in arrays, and
  closes a pack at its byte/count bound or a change of locality key. Numerical
  matrix data is not copied. Array snapshots are included in its reservation.
- A lane accepts one outstanding handoff. The returned future is backpressure:
  producers must await it before reusing that lane. Compute continuations do so
  asynchronously; the bounded synthetic source uses dedicated producer threads.
  There is no unbounded input queue and no global per-tile materializer lock.
- Open packs have explicitly admitted, bounded memory. They are not visible in
  the store and are not yet mutable, spillable cache entries. Each lane reserves
  space for an open pack, a detached handoff, and array-construction overhead.
- Descriptor registration uses a short store lock; cache publication and
  downstream notification are outside it. Completion/closure requires joined
  producers. Publication failures propagate to store completion.
- `PartitionedStore` now requires an explicit range-index choice. An unindexed
  store can use a size-only descriptor and enumerate/acquire partitions without
  constructing range arrays. Indexing remains useful for the contraction input.
- The diagnostic `PackedMatrixOps` projection has four independent asynchronous
  scan lanes. The contraction uses bounding-box candidates, then matches actual
  tile coordinates with a sorted merge. Neither requires aligned packs. The
  contraction currently has one outstanding pair task; it is not a tuned
  parallel contraction primitive. MatrixBlock multiplication kernels use one
  thread. Sorting scratch is covered by task admission.
- Neither the planner nor PageRank's source/lease path is changed. The previous
  contiguous-row prototype remains available; the new materializer does not
  impose its membership assumptions.

## Conditions

- Four source producers claim chunks of 17 row tiles and emit each chunk in
  reverse order. This is an explicit synthetic concurrent source, not a native
  file-reader throughput experiment. Projection output goes through the same
  packing utility, with arrivals determined by actual worker progress.
- Singleton control: one tile per PackedBlock, using the same operator code.
  This isolates grouping effects but is not an ordinary MaterializedStore tile
  baseline. It still has a pack wrapper and the new materializer's ownership path.
- Unordered: up to 32 members, 512 KiB cap, constant locality key.
- Locality-guided: the same caps, key `(rowTileIndex - 1) / 32`. A key can produce
  multiple partial packs in different lanes; there is no prescribed matching
  pack ID in the other input and no requirement that a window be complete.
- X tiles are about 16 KiB and Y tiles about 64 KiB. Consequently, byte-bounded
  Y packs generally contain fewer tiles than X packs. End packs and locality
  changes can produce smaller packs; there is no claim of optimal packing.
- JVM: 3 GiB maximum heap, ActiveProcessorCount=4, 30 seconds of warmup, three
  measured iterations. This is not a CPU quota. No forced cache flush between
  phases or iterations. The 64 MiB cache spills; the 1 GiB cache fits the 8M-row
  case. Cache eviction watermark is 75% of capacity.
- Read/write numbers are completed serialized cache I/O bytes, not cgroup
  io.stat or physical SSD traffic. OS page cache is not flushed. Asynchronous
  writes can cross phase boundaries. No cgroup limit is imposed in this
  component experiment (`systemd-run` is unavailable in this container).
  The fixture explicitly uses buffered I/O, not direct I/O.
- Preparation times include generation, packing and publication. Iteration
  times include output packing and admission/handoff waits, projection,
  contraction, validation and normalization, but exclude initial X generation,
  subsequent store teardown and final cache shutdown.
- Every iteration is checked against an independently computed 2x2 Gram matrix.
  The synthetic data uses a deterministic 97-row repeating pattern; timing is
  not a claim about convergence or a representative statistical dataset.

## Reproduction

Build in the SystemDS checkout:

```bash
mvn -q -Dtest=PackingMaterializerTest,PartitionedStoreTest,RowPackedMatrixOpsTest test
```

Run from this experiment repository (arguments: rows, cache bytes, maximum pack
members, locality-window tiles; zero window selects unordered):

```bash
JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  8000000 67108864 32 32 1 0
```

Use `1 0` for the singleton control, `32 0` for unordered packing, and cache
`1073741824` for the fitting-cache case. Source and intermediate spill files are
temporary and cleaned by cache shutdown. Raw logs are in data_dir under
`packing-bytecap-*`; the earlier `packing-8m-spill-*` logs used a tile-count-based
output size and are not part of the final byte-cap comparison.

## Results

Medians of the three iterations, including projection output packing. GB are
decimal; cache capacities are binary MiB/GiB. Full per-phase observations are in
`concurrent-packing-results.csv`.

| Rows / cache | Strategy | Projection + contraction (s) | Contraction reads (GB) | Executor dispatches / iteration |
|---|---|---:|---:|---:|
| 8M / 1 GiB | Singleton | 0.252 | 0 | 32,005 |
| 8M / 1 GiB | Unordered | 0.234 | 0 | 10,977 |
| 8M / 1 GiB | Locality-guided | 0.227 | 0 | 4,273 |
| 8M / 64 MiB | Singleton | 1.889 | 0.606 | 32,007 |
| 8M / 64 MiB | Unordered | 1.674 | 2.709 | 11,189 |
| 8M / 64 MiB | Locality-guided | 0.644 | 0.640 | 4,274 |
| 100M / 64 MiB | Singleton | 29.728 | 7.995 | 400,027 |
| 100M / 64 MiB | Unordered | 36.138 | 35.553 | 142,967 |
| 100M / 64 MiB | Locality-guided | 18.939 | 8.403 | 53,757 |

For 100M rows, X is 1.6 GB and Y is 6.4 GB. Source generation plus packing took
0.510 s (singleton), 0.385 s (unordered), and 0.670 s (locality). Thus locality
adds measurable preparation cost, rather than receiving a free prearranged
layout. Its median iteration cost was about 1.57x lower than singleton and
1.91x lower than unordered packing. At 8M rows, spilling improved about 2.93x
over singleton; the fitting-cache difference was only about 10%.

The amplification mechanism is visible in the candidate-task counts. At 100M
rows, unordered contraction schedules 65,802 candidate pairs per iteration,
versus about 17,857 for locality-guided packing. Bounding boxes for irregular
packs overlap even when many contained tile coordinates do not. Acquiring such
packs costs full-pack reads; the coordinate merge later discards nonmatches.
Reacquisition and eviction compound this. Locality constrains the overlap and
keeps reads near the approximately 8 GB logical contraction input. It does not
eliminate all repeated reads (8.40 GB was observed).

Writes do not show the corresponding amplification: median combined stage
writes at 100M rows were 6.399 GB, 6.394 GB and 6.396 GB, respectively, close to
one Y spill. The unordered regression is predominantly extra reads/decoding and
candidate processing, not repeatedly rewriting the same intermediate.

Packing locality trades fill for useful access. At 100M rows, unordered packing
created 3,128 X packs and 12,502 Y packs; locality produced 8,824 X packs and
about 16,802 Y packs. It was faster despite having more entries because it
substantially reduced unrelated pack access. Exact membership indexing and
predetermined matching partitions were not required for this improvement.

These are diagnostic measurements, not paper-ready confidence intervals: one
JVM per configuration, no randomized configuration order, no cgroup isolation,
and substantial per-iteration variance on the larger runs. For example, the
unordered 100M iteration times ranged from about 22.9 to 51.0 s. The repeatable
byte/task differences are stronger evidence than precise speedup estimates.

## Validation and profile

Four `PackingMaterializerTest`, four `PartitionedStoreTest`, and six
`RowPackedMatrixOpsTest` tests passed. New coverage includes concurrent,
unordered, iterative projection/contraction with actual spills, a partial final
tile, unindexed enumeration/reload, withholding open packs, oversized-input
rejection, and publication-error propagation with ownership cleanup.

A separate 100M-row locality run was profiled with JFR after JVM warmup; its
timings are not included in the comparison table. Recording started by
attachment after the measured-phase header, so it does not cover all initial
source generation. The 61-second recording contains 3,110 Java/native execution
samples. Leading top frames were native file reads (576), two matrix-multiply
vector kernels (437 and 311), buffered double-array serialization (358), native
file writes (292), matrix-multiply copying (285), and dense-block construction
(176). These are sampled stacks, not exact CPU-time attribution.

Only five sampled stacks contained the packing materializer and one contained
partition publication. Stack-depth limits and sampling mean these counts are
not a precise measurement of total packing cost. There were no JavaMonitorEnter
events at the profile's 10 ms threshold; this does not exclude shorter lock
contention. GC pauses totaled 0.709 seconds across 223 collections. There is no
evidence here that a global packing lock dominates; the implementation has no
such per-tile lock, and the large read/candidate-count difference provides a
more direct explanation for the unordered regression.

The recording and profile log remain in data_dir as
`packing-profile-100m-locality.jfr` and `packing-profile-100m-locality.log`.

## Production follow-up

The reusable boundary is the materializer, not the diagnostic algorithm driver.
Next integration should supply producer lanes from actual OOC streams, preserve
source-backed reload/live lease delivery, and derive pack bounds from supported
task/cache sizes. In particular, a pack must fit the cache's handoff capacity;
the old cache does not make progress if an individual entry exceeds that limit.
The initial tiny-cache test exposed that constraint; final forced-spill tests
use a cache that can hold individual packs but not the complete input/output.

Better locality policies can route related arrivals to common owners rather
than flushing producer-local builders on key changes. This should improve pack
fill without imposing a predetermined physical partition layout. It should be
measured before adding a more elaborate index or general repartitioning planner.
