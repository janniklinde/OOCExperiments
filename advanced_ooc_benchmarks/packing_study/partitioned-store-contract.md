# Partitioned store prototype

`PartitionedStore<D extends Descriptor>` owns closed `PackedBlock` entries through the existing materialized store and cache. Partitioning strategies construct packs outside the store and publish them only after closure. The implementation does not change global primitive selection or the existing source materializer.

```java
PrimitiveIterator.OfInt filter(long minRow, long maxRow, long minCol, long maxCol);
D describe(int partitionIndex);
OOCFuture<StoreLease<PackedBlock>> acquire(int partitionIndex, MemoryAllowance allowance);
```

- Bounds use zero-based matrix-cell coordinates: `[minRow, maxRow) × [minCol, maxCol)`. Empty query ranges match nothing.
- Partition indices are primitive `int` values assigned in publication order. They remain stable after index construction; they do not encode geometry.
- Descriptors are immutable and provide enclosing bounds, resident payload bytes, and descriptor memory accounting. `Bounds` is the basic rectangle descriptor; specialized descriptors may expose additional membership or geometry information. The publisher is responsible for truthful bounds and immutable payload ownership.
- `filter` is exhaustive only after successful materialization completion. It yields bounding-box candidates without reading payloads; consumers must examine actual members where an irregular pack's bounds contain holes. Iterator order should not be relied upon by general consumers.
- The initial directory uses primitive sorted partition indices and prefix maximum row ends. Lookup binary-searches the first possible row overlap and checks columns while traversing candidates. This is efficient for row-local packs; broad overlapping boxes may require scanning many candidates. No spatial tree is introduced.
- `subscribe(IntConsumer)` registers once before publication. It receives each newly published closed partition's index; `describe` and `acquire` are available at that point. It is a nonblocking notification, not a replay subscription or a lease transfer. `completion()` is the separate completion signal. Materializers must finish their publication calls before completing the store.
- The caller reserves whole-partition bytes before acquisition and keeps the lease until finished. Directory descriptors and index arrays are charged separately to the metadata allowance. Store closure releases that accounting.

`RowPackedStore` retains the original one-column-tile validation and row-pack descriptor as a specialization of the common store. `RowPackBuilder` still holds unpublished data until `finish()`. The projection/contraction prototype now uses primitive partition indices and rectangular filtering rather than a row-specific list of packs. Retiling and planner negotiation remain future work.

Validation: four `PartitionedStoreTest` tests plus six `RowPackedMatrixOpsTest` tests. These cover randomized rectangle lookup against brute force, exact boundaries, coordinates beyond double integer precision, stable IDs, closed-pack notifications, empty stores, metadata admission, real spills/reloads, unequal/shuffled row packs, and numerical/error-path checks.

The existing 8M × 2 projection/contraction experiment was repeated with the same 30-second JVM warmup and three contraction repetitions. Median contraction times were 0.1102 s versus 0.0826 s with the fitting cache, and 1.2938 s versus 0.2531 s with the spilling cache (single-tile versus 32-tile packs). The packed spilling advantage remains approximately 5.1×, reading 384.55 MB versus 384.62 MB. As before, the OS page cache was warm and both arms used the prototype, not the existing DML pipeline. Measurements are in `partitioned-store-benchmark.csv`.

```bash
cd /workspace/systemds
/opt/devcon/env/bin/mvn -q -Dtest=PartitionedStoreTest,RowPackedMatrixOpsTest test
```
