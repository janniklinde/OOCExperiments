# Twitter source-read investigation (2026-09-24)

## Scope and setup

Published Twitter vertex ordering: 41,652,230 vertices and 1,468,365,182 edges.
Native FP64 binary-block graph; complete tile grid, including empty tiles.
The three OOC-written variants have 16 files each, but very unequal file sizes.

| Block size | Logical tiles | Empty tiles | Native graph bytes (GB) | Largest file (GB) |
|---|---:|---:|---:|---:|
| 10,000 | 17,355,556 | 176,545 | 21.508 | 7.519 |
| 5,000 | 69,405,561 | 6,888,261 | 23.514 | 7.674 |
| 2,500 | 277,588,921 | 126,537,772 | 30.895 | 8.224 |

PageRank controls: 16 GiB cgroup, 12 GiB JVM heap, 16 logical processors,
16 source/replay readers unless stated, direct I/O, 1 GiB source phase and
replay limits, 16 MiB source partitions. Cache-wrapper packing remains off.
The graph is packed at the source/materialized-store boundary. Cold-cache
precondition checked. Read/write bytes come from cgroup v2 `io.stat`.
Wall times include JVM startup, initial scan, PageRank, and writing the vector;
they are NOT isolated source-read timings. Decimal GB throughout the tables.

Logs, configurations, telemetry, vectors and JFR profiles are under
`data_dir/pagerank-reader-study/`. Probe sources and runner are in this directory.
Runs were sequential, not concurrent benchmark workloads. Most comparisons are
single runs, with selected repeats; differences of a few percent are not conclusive.

## Experiment ladder: storage, decoding, concurrency, end-to-end

1. Raw direct reads of the same 2.5k graph, 16 readers, 512 MiB byte ranges:
   5.796 s / 5.33 GB/s; repeat 6.052 s / 5.11 GB/s. Storage can approach the
   stated 5.5 GB/s ceiling. These are standalone read probes, not PageRank runs.
2. SequenceFile + MatrixBlock decode only, without cache, phase accounting or
   PageRank: the 2.5k graph takes 16.063 s with one task/file, or 12.701 s with
   512 MiB sync-aligned splits. The 5k graph improves from 11.085 s (repeat
   11.250 s) to 6.401 s; the 10k graph from 10.450 s to 5.626 s.
   Exact record/empty counts match between unsplit and split scans.
3. Actual PageRank source path: vary buffer size and reader concurrency, then
   profile monitor contention and change only grouped phase accounting.
4. Validate end-to-end behavior with the full 15 iterations, not only startup.

The split-reader prototype uses SequenceFile sync markers and normal boundary
semantics, so no external index is required to split a full scan. Production
SourceStore still tracks cursors/layouts per file; within-file splitting is NOT
integrated. It requires per-split continuations and ordered layout merging.

## Verified physical overread

| One-iteration case, 16 readers | 8 MiB buffer read GB | 1 MiB buffer read GB | Wall 8 / 1 MiB (s) |
|---|---:|---:|---:|
| 5k | 32.692 | 24.710 | 26.826 / 26.526 |
| 2.5k | 48.400 | 33.229 | 84.232 / 81.434 |

Bounded source phases reopen files. Buffered bytes beyond the stopping position
are read again in a later phase. This is real physical overread, not an io.stat
unit error. A smaller buffer substantially reduces bytes, but barely changes
runtime: disk bandwidth is not the dominant limit in these cases.
The PageRank experiment plan now uses a 1 MiB reader buffer.

The original 10k layout with 4,166 files is particularly bad with bounded phases:
102.644 GB read and 27.672 s, versus 22.981 GB and 17.072 s for the new 16-file
layout. In a standalone unbounded decode, however, the many-file layout takes
only 6.724 s. File count alone is therefore not the explanation; bounded reopening
and the file-size distribution matter.

## Verified phase-budget contention and implemented fix

Before the fix, every decoded tile entered the same SourceStore phase-budget
monitor and updated shared counters. At 2.5k that means approximately 278 million
monitor acquisitions. Two thread dumps found 14/16 and 11/16 source readers
blocked on that monitor. JFR (monitor threshold 1 ms) measured 58.1 blocked
thread-seconds there. This excludes shorter waits and is not wall time.

SourceStore now leases capacity in bounded 64 KiB credits per grouped reader,
debits tile memory locally, aggregates serialized-byte statistics locally, and
returns unused credit on exit. It still charges actual tile + pack metadata,
enforces the shared phase limit, and preserves single-oversized-record progress.
Unused credit is bounded by 64 KiB per reader (less than 1 MiB for 16 readers).
The ungrouped/preflight path is unchanged. This is not unbounded read-ahead.

| 2.5k, one iteration, 16 readers | Wall (s) | CPU-seconds | Read GB |
|---|---:|---:|---:|
| Original, no profiler | 81.434 | 724.49 | 33.229 |
| Credit batching, no profiler | 49.278 | 325.64 | 32.980 |
| Original, JFR | 84.049 | 737.05 | 33.237 |
| Credit batching, JFR | 50.531 | 321.59 | 32.974 |

One-iteration regression checks at larger tile sizes:

| Block size | Original wall (s) | Credit batching wall (s) | Original / new CPU-seconds |
|---|---:|---:|---:|
| 5,000 | 26.526 | 25.024 | 226.89 / 174.19 |
| 10,000 | 17.072 | 17.323 | 126.23 / 128.13 |

No meaningful 10k change is established by this single pair; the 2.5k benefit is
much larger and reproduced with/without profiling. All candidates completed
without cgroup OOM events.

That is a reproducible ~39% wall-time reduction without changing the dataset,
algorithm, packing size, reader count, or memory limit. The source monitor no
longer appears among measured contended monitors after the fix. GC pauses were
1.121 s before / 1.352 s after: stop-the-world GC does not explain the difference.

Eight SourceBackedReadOOCIOHandler tests pass, including a new grouped-credit
test checking tight limits, oversized-record progress, exact serialized-byte
counts, no duplicate/missing tiles, and value correctness.

The full 15-iteration 2.5k run also improves: 301.240 -> 253.884 s (15.7%),
CPU 3883.00 -> 3432.62 seconds, physical reads 453.232 -> 454.700 GB, writes
1.586 -> 1.785 GB. Peak cgroup memory after the fix is 15.306 GB, with no OOM
events. This is primarily a startup coordination improvement, not an iteration
read-volume reduction. Internal `loadFromDisk` GB are not substituted for
physical `io.stat` bytes in these comparisons.

CP comparison of retained reference/candidate vectors gives maximum absolute
errors 3.25e-18 (2.5k, one iteration), 1.32e-17 (2.5k, 15 iterations),
5.42e-19 (5k), and 2.60e-18 (10k). Rank sums remain approximately
1.000000000000599. Validation commands use `compare_pagerank_vectors.dml`;
each candidate directory contains `validation.log`.

Reader-count control on the original 2.5k path, one iteration:

| Readers | Wall (s) | CPU-seconds |
|---|---:|---:|
| 2 | 84.832 | 236.10 |
| 4 | 57.728 | 270.25 |
| 8 | 66.279 | 509.50 |
| 16 | 81.434 | 724.49 |

This non-monotonic result motivated the contention profile. Simply setting four
readers is not a general fix: over 15 iterations, the original 4-reader run takes
323.394 s versus 301.240 s with 16, since the setting also restricts replay reads.

## Publication follow-up: recorded sizes and concurrent handoff

The follow-up implements the first two recommendations below. Source groups now
carry a primitive `long[]` of tile memory charges plus the already computed pack
charge. The source primitive passes these through to ordinary/partitioned store
publication without resumming MatrixBlock sizes or creating one callback per tile.
The final PackedBlock retains its existing ownership/serialization representation.

Partitioned publication no longer holds a monitor while building a pack, inserting
it into the cache, or calling the live consumer. An atomic counter assigns unique
partition IDs and an atomic terminal flag handles completion/failure. Cache/store
operations retain their own synchronization. SubscribableTaskQueue emits EOS only
after all active deliveries return; source-phase budgets and downstream partition
slots still bound admission. There is no new asynchronous queue or unbounded producer.

The 31 focused tests pass (19 MaterializedStore, 9 SourceBackedReadOOCIOHandler,
3 SourceReadOOCIOHandler). New checks explicitly block one live handoff while
allowing another to publish, verify EOS waits for the blocked delivery, check
ownership cleanup on failure, and use MatrixBlocks whose size methods throw to
prove ordinary and partitioned publication reuse the recorded charges. Group
failure propagation and carried-size equality are also checked.

One-iteration, unprofiled PageRank (same configuration as above):

| Block size | Credits-only wall (s) | New wall (s) | CPU-seconds before / after | Read GB before / after |
|---|---:|---:|---:|---:|
| 2,500 | 49.278 | 29.479 | 325.64 / 370.54 | 32.980 / 33.220 |
| 5,000 | 25.024 | 19.672 | 174.19 / 202.35 | 24.790 / 25.804 |
| 10,000 | 17.323 | 16.522 | 128.13 / 136.54 | 22.695 / 23.461 |

The benefit is reduced elapsed time, not reduced total CPU or physical I/O.
More concurrent publication changes arrival order and source-phase boundaries;
reads increase slightly. Do not claim the small 10k difference is significant
from a single pair. No candidate reports cgroup OOM events.

The profiled 2.5k run changes from 50.531 to 32.030 s (CPU 321.59 to 390.75 s).
The publication-monitor events disappear. The largest remaining sampled monitor
waits are MaterializedCallback.get (4.47 thread-seconds), sparse matvec accumulator
access (2.42), and cache insertion (2.29). GC pauses total 1.197 s. These are
overlapping thread-seconds, not additive wall-time components.

The three unprofiled one-iteration vectors agree with the retained pre-change
2.5k vector to at most 4.49e-17 absolute error; rank sums remain approximately one.
Detailed logs/profiles are `ooc{2500,5000,10000}-r16-publication` and
`ooc2500-r16-publication-profile` under the same diagnostic results root.

The full 15-iteration follow-up (`ooc2500-r16-publication-iter15`) completes in
241.742 s versus 253.884 s with credits alone (~4.8% faster in this single pair).
CPU rises from 3432.62 to 3595.24 seconds. Reads are essentially unchanged,
454.700 -> 454.918 GB; writes are 1.785 -> 1.980 GB. Peak cgroup memory is
15.527 GB, with zero OOM or memory-max events. These changes primarily accelerate
initial source publication; they do not optimize the repeated matvec/replay kernel.
The final vector differs from the credits-only reference by at most 3.04e-18;
both rank sums are 1.0000000000005989.

The final local JAR was rebuilt from SystemDS base
`a78355f99eed3e88fbd04dbfd2c6ff552a8effc5` plus this publication/sizing diff;
SHA-256 `292fd289c619d0b3db9e0ff1c160e3e6ee147e6e7336b9ad01db704c79f9864d`.
The reader buffer remains 1 MiB; no reader-lifetime, file-splitting, or global
packing configuration changes were made in this follow-up.

## Remaining costs and next steps (original diagnosis)

After the source-budget fix, JFR shows 342.6 blocked thread-seconds at
PartitionedOOCStreamMaterializer.accept, up from 58.8. This synchronized method
does memory sizing, cache publication and invokes the live consumer, which may
wait for matvec partition slots. This is a mixture of serialized publication
and downstream backpressure; it is NOT evidence of 342 seconds spent on storage.

CPU samples identify matvec partition processing, per-tile callback access,
column-set construction, MatrixBlock decode, and repeated memory/serialized-size
calculations. Some latter stacks recount nonzeros during source handoff and
materialization. These are concrete follow-up targets, not yet isolated A/B fixes.

Recommendations from the first study (the first two are implemented above):

- Carry already computed memory charges through immutable source groups, avoiding
  repeated sizing passes at handoff/publication.
- Narrow partition publication synchronization while preserving publication/EOS
  ordering, failure handling, and bounded downstream admission. Do not simply
  remove backpressure or the monitor.
- Integrate sync-aligned within-file tasks after the above, retaining bounded
  source admission and correct per-split continuations. The isolated split probe
  establishes feasibility and the benefit of balancing skewed source files.

Conclusion: "small-tile overhead" is too imprecise. A major avoidable component
was shared per-record accounting; another verified component is bounded-buffer
overread. Remaining decoding, publication and algorithm work means a PageRank
wall-time-derived GB/s should not be expected to equal raw SSD bandwidth.

## Preparation changes and limitations

The experiment plan now uses `real_world/prepare_pagerank_ooc_variant.sh` and
`G-ooc-bs{2500,5000,10000}` / corresponding dangling-vector datasets. Preparation
uses a 20 GiB heap and 16 workers. `sysds.ooc.write.empty.blocks` explicitly
controls OOC binary output; default `true`, independent of the COO allocation
setting. Keep it true for these complete-grid experiments.

Preparation currently creates target-sized tiles from CSR and rewrites them
through the OOC native writer to establish the new file layout. This is a hybrid
path, not an efficient generic OOC tile-size conversion: the actual 10k-to-5k OOC
reblock was too slow due to fragment processing. At 2.5k, preparation enables
temporary cache packing to avoid metadata pressure; this does not enable that
cache wrapper for the benchmark itself.

Stale direct-converter 2.5k/5k staging graphs and dangling vectors were deleted
after replacement validation, freeing roughly 51 GiB. They can be recreated from
the retained CSR input. Original 10k input and the successful OOC variants remain.
