# Aligned, pre-partitioned control

This follow-up keeps the previous 100M x 2 FP64 workload, 1000-row logical tiles,
2 x 8 resident P, 100M x 8 intermediate Y, 64 MiB OOC cache, 3 GiB JVM heap,
four projection lanes, and single-lane contraction. The Java kernels and generic
partition lookup/acquisition path are unchanged. Each JVM gets 30 seconds of
warmup and three measured iterations.

The ideal control prepares contiguous X packs directly, outside the operator
timers. Each contains 32 row tiles. Y is produced in aligned eight-tile packs
under the same 512 KiB cap. Every Y pack lies wholly inside one X pack; there
are no irrelevant candidate pairs or holes. Alignment and complete row coverage
are checked, including a partial final tile in the forced-spill test.

This is perfect logical partitioning, not a new contiguous file format. It uses
the existing spill-file layout and reader. The contraction timer sees both
inputs already materialized: no source preparation or Y packing is timed there.
The full iteration measurement still includes computing/packing Y, since it
must change on each iteration. Alignment validation is included in the measured
projection phase of the ideal control, rather than giving that control an
unreported timing advantage.

The sequence is ideal A, fresh locality-guided control, ideal B, in separate
JVMs. Buffered I/O and a warm OS page cache remain intentional, matching the
previous experiment. New counters record `/proc/self/io` read_bytes/write_bytes
deltas per stage in addition to serialized cache I/O. The process read counter
distinguishes storage fetches from reads served by the OS page cache; the write
counter measures process-accounted writes, not the completion time of physical
SSD writes.

## Reproduction

```bash
cd /workspace/systemds
/opt/devcon/env/bin/mvn -q -Dtest=PackingMaterializerTest#alignedIterativeSpill test
cd /workspace/OOCExperiments
JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  100000000 67108864 32 -1 1 0
```

The fourth argument `-1` selects this test-only aligned-data preparation path;
the fifth argument `1` preserves the original serial contraction control.
`32` selects the earlier locality-guided path. No production primitive, planner
selection, or benchmark-plan configuration was changed for this comparison.
Raw logs are in data_dir under `packing-aligned-100m-*.log`.

## Results

| Layout / cache | Projection median (s) | Contraction median (s) | Combined iteration median (s) | Contraction cache reads (GB) |
|---|---:|---:|---:|---:|
| Locality-guided / 64 MiB | 4.075 | 15.149 | 19.108 | 8.441 |
| Perfect alignment / 64 MiB | 3.175 | 11.974 | 15.046 | 8.008 |
| Perfect alignment / resident | 1.503 | 3.530 | 4.672 | 0 |

The ideal result pools six iterations from two JVMs; the other rows each have
three iterations. Component medians need not sum to the median combined time.
The two ideal JVMs separately had combined medians 14.466 s and 15.625 s, and
contraction medians 11.722 s and 12.019 s. All numerical checks passed. Initial X
preparation took 0.614/0.617 s for ideal A/B, versus 0.640 s for locality; this is
excluded from the operator timings. Full observations are in
`perfect-partitioning-results.csv`.

Perfect logical partitioning reduced combined time about 21% relative to the
fresh locality control, or about 1.27x faster. It eliminated unnecessary range
candidates: 12,500 pair tasks (25,001 executor dispatches including continuations)
per contraction instead of about 17,900 pair tasks for locality-guided packs.

### Why this is not an SSD bandwidth result

Every measured projection and contraction in the spilling ideal and locality
runs had **zero process storage-read bytes**. The serialized cache reads came
from the Linux page cache. Thus, dividing 8 GB by 12 seconds gives the throughput
of the complete cache-load/deserialization/compute path, not SSD throughput.
The user's 5.5 GB/s storage rate would imply approximately 1.45 seconds for a
pure sequential 8 GB read, but this experiment did not issue that amount of
storage reads and includes substantial work beyond reading bytes.

The resident control uses the same ideal packs, kernels and concurrency, but a
12 GiB cache and 12 GiB maximum JVM heap. Available host memory was checked
before starting it; the data payload is about 8 GB. It recorded zero cache reads
and writes. Its contraction time was 3.53 s, versus 11.97 s in the spilling ideal
case; median contraction CPU time was 4.22 s versus 15.08 s. That gap belongs to
the additional spill/reload, deserialization/allocation, and cache-management
path. It is not an isolated measurement of any one of those components.

The generic contraction in this experiment had only one pack-pair task in flight. Each
pair is acquired, processed tile by tile, and released before the next pair is
scheduled. Perfect partitioning does not remove that restriction or replace the
100,000 small MatrixBlock multiplication/transpose operations with larger
kernels. Consequently these numbers are not a hardware-saturating ceiling for
an optimized partition-aware implementation. Overlapping/parallelizing pair
processing and isolating spill-reader decoding costs are the next relevant
investigations; better grouping alone cannot explain or remove the remaining
gap demonstrated here.

Resident-control reproduction:

```bash
PACKING_HEAP=12g JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  100000000 12884901888 32 -1 1 0
```

The focused `PackingMaterializerTest#alignedIterativeSpill` test passed with a
2 MiB cache, actual spill/read activity, and a partial last tile. The experiment
remains a component workload, with buffered I/O and no cgroup isolation, not a
paper-ready full-system comparison. No production implementation was changed.
