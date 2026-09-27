# Bounded atomic match reservations

The opt-in PackedMatrixOps.crossProduct prototype now accepts maxAtomicBytes.
A positive value selects bounded match batching; zero retains the pair-scoped
control. Normal DML primitive selection is unchanged.

For each left partition, a lane admits enough for that partition, the largest
single right partition, and scratch. It acquires and retains the left lease,
then selects as many right matches as fit under the cap. It tries to grow its
reservation without waiting. On denial or reaching the cap, it requests and
processes the selected matches, releases their leases, and reuses the reservation
for the next batch. The left lease survives all these batches. Only after the
entire match set finishes does the lane release it and the reservation.

The cap covers inputs and scratch; the small lane-local result accumulators are
separately reserved for the contraction's duration. The cap must fit a maximum
single pair. No additional parent admission is awaited while holding a lease.
The existing per-pair kernels are unchanged; left-tile sorting is now reused
across matches. Parallel lanes retain separate output accumulators.

ReservationBudget.tryGrow(bytes) explicitly adds available capacity through the
parent's nonblocking memory reservation, without applying new-task admission.
The older enableGrowth/tryReserve behavior is unchanged for existing callers.
This distinction mattered experimentally: the old growth path applies
SyncMemoryAllowance's active-task heuristic to a small growth increment and
rejected nearly all increments with eight active lanes, even with free memory.
It therefore retained X but did not actually batch its four matches.

## Experiment

Same aligned 100M x 2 FP64 X, 100M x 8 Y, blocksize 1000, 64 MiB cache,
3 GiB heap, eight contraction lanes, four compute threads and additional I/O
threads. Each separate JVM warms up for 30 seconds and measures three iterations
of Y = X P; P = X^T Y followed by normalization. Every result is numerically
checked. Times below are contraction-only medians. Decimal GB for I/O.

| Mode | Atomic cap | Time (s) | CPU (s) | X reload (GB) | Y reload (GB) | Total reload (GB) | Executor dispatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Release both after each pair | control | 1.745 | 15.87 | 3.685 | 6.395 | 10.080 | 25,008 |
| Retain X, one Y at a time | 2,300,000 B | 1.847 | 12.42 | 1.584 | 6.395 | 7.979 | 31,258 |
| Retain X, batch all four Y matches | 4 MiB | 1.603 | 13.37 | 1.588 | 6.394 | 7.983 | 12,508 |

The bounded batch reduces total reload by about 21%, halves executor dispatches,
and is about 8% faster than the pair-scoped control. Relative to retained-X serial
processing it is about 13% faster. The timing improvement is modest, not an
order-of-magnitude change; independent JVM repetitions would be needed for a
precise performance claim. The read-volume result directly confirms that the
earlier amplification was repeated X retrieval: Y was already read once.

All contraction stages in these three selected runs have zero process storage
read_bytes. Reloads are served by the OS page cache; this does not establish SSD
bandwidth. No cgroup isolation or explicit cold-cache reset was used. Projection
times fluctuate substantially, so full-iteration timing is not used to attribute
this contraction optimization.

Raw logs are data_dir/packing-atomic-100m-{pair8,grown8,retain8-final}.log.
Also preserved: retain8 (initial serial-retention run, including 0.18/0.64 GB of
physical reads in its first two contractions) and batch8 (the initial 4 MiB run
whose growth was prevented by new-task admission). These preliminary runs are
not pooled into the table. All observations and per-input read counters are in
atomic-packing-results.csv.

## Validation and reproduction

Nineteen focused tests passed: PackingMaterializerTest (7), PartitionedStoreTest
(4), RowPackedMatrixOpsTest (6), and two selected OOCMemoryAllowanceTest methods.
Coverage includes actual spill/reload, numerical comparisons, partial final
tiles, unaligned partitions, eight lanes sharing a 3 MiB allowance despite a
4 MiB desired atomic cap, the one-match cap, read failure, executor rejection,
and explicit budget growth without task readmission. Reservations are checked
for cleanup. No full suite or full-algorithm DML benchmark was run.

```bash
cd /workspace/systemds
/opt/devcon/env/bin/mvn -q \
  '-Dtest=PackingMaterializerTest,PartitionedStoreTest,RowPackedMatrixOpsTest,OOCMemoryAllowanceTest#testExplicitBudgetGrowthDoesNotReadmitTask+testGrowingReusableBudget' test
cd /workspace/OOCExperiments
JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  100000000 67108864 32 -1 8 4194304
```

The sixth argument is maxAtomicBytes. Use 0 for pair-scoped processing or
2300000 for retaining X with one right match at a time. Runner defaults are
four lanes and a 4 MiB cap. Older experiment reproduction commands explicitly
select a zero cap to preserve their original behavior.
