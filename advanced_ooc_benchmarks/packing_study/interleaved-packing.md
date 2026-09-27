# Interleaving partition-pair retrieval and computation

The opt-in `PackedMatrixOps.crossProduct` prototype now accepts an explicit
concurrency limit. Independent asynchronous lanes claim left partitions, walk
their matching right partitions, and accumulate into small lane-local matrices.
Each pair reserves its complete working set before acquiring either input.
Retrieval, computation, and asynchronous release can overlap across lanes;
there is no blocking wait on a compute thread. Lane accumulators are reserved
up front and merged only after every lane finishes. Failure also waits for all
lanes before releasing their accumulator reservation.

This changes neither partition layout nor per-tile arithmetic. It does not
enable packing or change primitive selection in normal DML execution. The
experiment runner defaults to four lanes; its fifth argument selects another
limit. Earlier experiment commands now explicitly select one lane.

## Setup

Same component workload as the aligned-partition experiment:
`Y = X %*% P; P = t(X) %*% Y`, followed by normalization of P.
X is 100M x 2 FP64 (1.6 GB), Y is 100M x 8 (6.4 GB), P is 2 x 8.
Logical blocksize is 1000; aligned X packs contain 32 tiles and Y packs eight
tiles, under a 512 KiB pack cap. Both contraction inputs are materialized.
The cache is 64 MiB, heap 3 GiB, with four compute threads and
ActiveProcessorCount=4. I/O threads are additional; this is not a four-core
cgroup quota. Each separate JVM gets 30 seconds of warmup and three measured
iterations. All results are checked against an independently computed Gram
matrix. X preparation is excluded from the operator timings.

Runs were sequential: aligned concurrency 1, 4, 8, then locality-guided 4.
Buffered I/O, no explicit page-cache drop, no cgroup isolation.

## Results

Medians over three iterations; GB means decimal GB.

| Layout | In-flight pairs | Contraction elapsed (s) | Contraction CPU (s) | Cache reload (GB) | Full iteration (s) |
|---|---:|---:|---:|---:|---:|
| Aligned | 1 | 12.707 | 16.68 | 8.019 | 14.633 |
| Aligned | 4 | 2.598 | 11.94 | 8.156 | 4.917 |
| Aligned | 8 | 1.737 | 14.85 | 9.992 | 4.241 |
| Locality-guided | 4 | 3.369 | 16.10 | 8.206 | 8.544 |

Four lanes speed up the contraction 4.89x and the complete iteration 2.98x
relative to the fresh serial control. Eight lanes reach 7.31x contraction
speedup but reload about 25% more bytes than serial, versus about 2% more with
four lanes. Greater concurrency increases the active footprint; the additional
reloads are consistent with cache pressure, but individual eviction decisions
were not traced in this experiment. Four lanes are the conservative default.

All measured stages report zero `/proc/self/io` read_bytes. These are real
cache spill-file reloads served by the OS page cache, not physical SSD read
throughput measurements. Interleaving removes a major serialization bottleneck
in the complete reload/deserialize/compute/release path; it does not isolate
cache lookup cost from those other components. The previous comparison to a
resident serial contraction therefore overstated what could be attributed to
unavoidable cache overhead.

Projection was unchanged and shows timing variability (including a 6.06-second
first projection in the eight-lane run). Treat full-iteration figures as a
small component experiment, not a paper-ready full-algorithm result. The
locality-guided run confirms correctness/performance beyond perfect alignment;
it has no fresh serial locality control in this set.

## Validation and reproduction

Sixteen focused tests passed across PackingMaterializerTest,
PartitionedStoreTest, and RowPackedMatrixOpsTest. Coverage includes actual
spill/reload, partial final tiles, aligned and unaligned partitions, four/eight
contraction lanes, read failure and executor rejection with reservation cleanup.

```bash
cd /workspace/systemds
/opt/devcon/env/bin/mvn -q \
  -Dtest=PackingMaterializerTest,PartitionedStoreTest,RowPackedMatrixOpsTest test
cd /workspace/OOCExperiments
JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  100000000 67108864 32 -1 4 0
```

Change the fifth argument to 1 or 8; change the fourth to 32 for locality-guided
packing. Raw logs: `data_dir/packing-interleaved-100m-{1,4,8,locality4}.log`.
All observations are preserved in `interleaved-packing-results.csv`.
