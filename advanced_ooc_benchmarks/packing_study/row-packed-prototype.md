# Row-packed projection and contraction prototype — 2026-09-25

This is an explicitly invoked component prototype, not a planner-enabled DML implementation.
It implements `Y = X B; Z = t(X) Y` through the real OOC cache and spill store.
No benchmark-plan settings or existing primitive selection were changed.

## Implementation

- `RowPackedStore`: existing `MaterializedStore<PackedBlock>` plus one accounted row-range descriptor per pack. Publication may be shuffled; descriptors are sorted on completion. Packs contain consecutive row tiles on a single-column-tile grid. Missing ranges represent zero tiles. Overlapping ranges are rejected.
- `RowPackBuilder`: bounded append-only construction under an already admitted budget. Closing transfers immutable payload ownership. It checks byte, tile-count, and adjacency limits. The producer must retry a rejected tile in another builder or handle an oversized tile explicitly.
- `RowPackedMatrixOps`: asynchronous projection and cross-product, one work admission per physical pack intersection. Different boundaries are matched by logical indices. Self-products pin each shared physical pack once. Kernels use existing matrix multiplication and addition operations.
- A cache progress correction makes eviction account for space required by a deferred unpin, even when existing cache contents are below the normal eviction threshold.

The small projection operand / contraction output stays reserved while work executes. Input I/O and ownership transfers are asynchronous. The prototype processes one pack intersection at a time; it does not yet parallelize intersections or negotiate layouts with the planner. It requires completed materialized input stores. The builder is also used for projected intermediates, but is not wired into the production source stream.

## Validation

`RowPackedMatrixOpsTest`: 6 passing tests covering unequal/shuffled pack boundaries, sparse data and omitted zero tiles, a partial final tile, real forced spills/reloads, projection followed by contraction, self-product, admission rejection, rejected executor, and injected read failure. Results match ordinary in-memory multiplication within relative/absolute tolerance `1e-9` and working reservations return to zero.

`OOCCacheImplTest`: 21 passing tests, including a new deterministic regression for a deferred handoff below the normal eviction threshold.

## Experiment

- X: 8,000,000 × 2 FP64 (128 MB of values); B: 2 × 4; Y: 8,000,000 × 4 (256 MB); Z: 2 × 4.
- Logical blocksize 1,000. X tiles contain approximately 16 KB of values; Y tiles approximately 32 KB.
- Control: one tile per pack. Experimental: 32 X tiles per pack; Y packs split at 17 tiles, yielding 17/15 boundaries within each full X pack.
- Both arms use the same prototype, kernels and single compute executor. This controls for physical granularity but is not a direct timing comparison against the existing DML primitive pipeline.
- Cache: 1 GiB (fitting) or 8 MiB (spilling); JVM heap 2 GiB, ActiveProcessorCount=4, single-thread multiplication kernels. Per-operation allowance target 64 MiB; descriptor allowance is separately accounted.
- 30 seconds of warmup in the same JVM, then three contraction repetitions. Projection measured once per configuration.
- Before each spilling contraction, the OOC cache is drained. OS page cache is **not** dropped. Thus these measurements exercise real serialization/file I/O but do not measure cold SSD throughput. No cgroup isolation.
- I/O is serialized read/write bytes recorded by the OOC spill store, not `io.stat`. CPU is process CPU time, including I/O threads and GC. `tasks` includes compute work and the scan continuations; approximately half are compute tasks. `peak_work_bytes` samples admitted working memory at task submission, not JVM RSS. Cache metadata is the end-of-phase accounting snapshot; directory bytes are separately charged.

| Metric | One tile/pack | 32 X tiles/pack |
|---|---:|---:|
| Fitting contraction, median | 0.1134 s | 0.0850 s |
| Spilling contraction, median | 1.5339 s | 0.2741 s |
| Spilling contraction CPU, median | 2.86 s | 0.43 s |
| Spilling projection, single measurement | 1.6279 s | 0.2422 s |
| Contraction compute tasks | 8,000 | 500 |
| Contraction admissions, including output | 8,001 | 501 |
| Spilling contraction reads, median | 384.624 MB | 385.574 MB |
| Contraction writes | 0 | 0 |
| Peak admitted contraction working bytes | 81,592 | 1,100,496 |
| Fitting cache metadata after contraction | 8,192,000 B | 384,000 B |
| Pack directory memory | 1,536,000 B | 72,000 B |

The contraction improves 1.33× in memory and 5.60× while spilling, with essentially unchanged read volume. This supports reducing physical entry/task granularity; it does not establish whether cache lookup, reservation, file-read submission, or deserialization dominates individually. Rare packed rereads add less than 0.5% in these repetitions. Larger batches increase the admitted working set while reducing task and metadata counts.

## Reproduction

From `/workspace/systemds`:

```bash
/opt/devcon/env/bin/mvn -q -Dtest=RowPackedMatrixOpsTest,OOCCacheImplTest test
/opt/devcon/env/java/current/bin/java --add-modules=jdk.incubator.vector \
  -Xms256m -Xmx2g -XX:ActiveProcessorCount=4 \
  -cp 'target/test-classes:target/classes:target/lib/*' \
  org.apache.sysds.test.component.ooc.RowPackedMatrixOpsTest 8000000
```

Raw measurements: `row-packed-prototype.csv`. Runtime logs also reside in `/workspace/data_dir/row-packed-test.log` and `/workspace/data_dir/row-packed-benchmark.log`. Temporary spills use `../data_dir` and are cleaned up by the cache.

Next integration step: adapt these bounded operations to the existing primitive scheduler and packed source stream. Keep layout negotiation and general multi-column-tile contraction outside this prototype until that integration is validated.
