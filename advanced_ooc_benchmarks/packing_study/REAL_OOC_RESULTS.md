# Real SystemDS OOC follow-up — 2026-09-19

This follows [FINDINGS.md](FINDINGS.md). The earlier Java pack prototype did
not execute SystemDS primitives. These controls **do** run the current
SystemDS OOC engine, but compare logical block sizes rather than a new
transparent physical pack. They establish where packing and pack-aware work
could pay off; they are not a claimed packing speedup.

## Controlled setup

The same seeded nonnegative FP64 matrix, 20M × 10 (1.6 GB decimal), was
prepared as row-major SystemDS binary blocks in 22 contiguous SequenceFile
parts. The 100-row variant has 200,000 logical 100 × 10 tiles of about 8 KB;
the 1,000-row variant has 20,000 logical 1,000 × 10 tiles of about 80 KB.
Both variants contain identical values. Each run used a 6 GiB cgroup-v2
limit, 4 GiB JVM heap, eight active processors and source readers, direct
I/O, 8 MiB configured reader buffer, 64 MiB source production phase, 256 MiB
broker cap, and 0.04/0.06 cache soft/hard fractions. The named input was
evicted from page cache before each run. Physical bytes below are cgroup
`io.stat` deltas. KMeans uses two fixed iterations; GNMF uses rank four and
one update. Both write full outputs. The isolated kernels also write their
full result and then checksum it. The run harness cleans its own temporary
and output files while retaining logs and metrics.

The native inputs and per-run evidence are in
`/workspace/data_dir/packing-e2e-20260919/`. The runnable focused probe is
[run_systemds_probe.py](run_systemds_probe.py); the KMeans/GNMF cases use the
existing `trace_amplification.py` study runner. Times are one run per case,
so the ratios are diagnostic, not paper-ready confidence intervals.

| Workload | 80 KB tile wall | 8 KB tile wall | Physical read, 80 → 8 KB | OOC get calls, 80 → 8 KB |
|---|---:|---:|---:|---:|
| One-pass `sum(X > 0.01)` | 2.07 s | 4.22 s | 1.77 → 3.22 GB | no cache gets |
| `X %*% B`, `B` is 10 × 8 | 5.82 s | 10.92 s | 2.96 → 5.11 GB | 40k → 400k |
| `X * rowSums(X)` | 8.17 s | 17.62 s | 3.29 → 5.45 GB | 60k → 600k |
| `X * (X > 0.5)` | 9.62 s | 12.02 s | 4.61 → 5.46 GB | 60k → 600k |
| KMeans, two updates | 6.27 s | 24.52 s | 7.61 → 12.49 GB | 540k → 5.40M |
| GNMF, rank 4, one update | 7.02 s | 49.17 s | 8.31 → 15.20 GB | 303k → 3.00M |

All runs completed. The two KMeans inertias match exactly, GNMF W/H checksums
match to FP rounding, and each isolated kernel's checksum matches across the
two block sizes. The one-pass control has no OOC spill or reload at either
block size: the excess read is already in the source path. In KMeans, logical
requested reload volume is 5.54 versus 5.90 GB and spill writes 1.16 versus
1.32 GB; in GNMF, requested reload volume is 6.04 versus 6.53 GB and writes
2.17 versus 2.83 GB. The much larger wall and physical-read differences
therefore are not explained by an extra full logical pass over X. Read-request
counts rise roughly tenfold (KMeans 76,747 → 847,907; GNMF 128,182 →
1,374,324), as do cache get/put calls.

## Source I/O is not the raw-file prototype

SystemDS's initial source scan reads one SequenceFile record per tile and
admits only a bounded bulk at a time. Its `OOCDirectInputStream` allocates a
buffer but normally reads only the requested aligned extent; `SourceStore`
also performs a preflight header read per record when a phase has a byte
limit. At 8 KB tiles, a 1.6 GB one-pass source-only control physically read
3.22 GB. Disabling direct I/O for **only this scan** read 1.61 GB and took
2.52 s, but it is not a valid full-algorithm control: non-direct source
reloads enable lookahead, and an attempted 8 KB KMeans run timed out.

An experimental change to fill more of the direct reader's buffer was built
and tested, then **reverted**. With the 8 KB source-only scan, configured
prefetch sizes of 64 KiB, 512 KiB, 2 MiB, and 8 MiB yielded respectively
2.57/3.24, 2.27/3.44, 2.27/4.11, and 2.92/6.76 (seconds/GB physically
read). The large-buffer overread is consistent with reopening 22 source
parts at bounded 64 MiB phase boundaries; it should not be labeled a
measured physical-pack effect. The 512 KiB version made full KMeans no
faster (24.52 → 24.22 s) and slightly increased physical read (12.49 →
12.73 GB). The reader was restored and the JAR rebuilt. Thus simple read
buffering cannot substitute for a physical pack that survives source reload,
nor can a pack ignore phase/admission boundaries.

## Where pack-aware primitives could help

- **Tall-skinny matmul:** the current `GeneralMMultOOCPrimitive` admits a
  pair, computes a partial, and emits/reduces that partial per logical tile.
  `X %*% B` with a small B can process successive row tiles under one
  physical pack lease/task, but still produces one output tile per row band
  unless output packing is also supported. For `t(W) %*% X` in GNMF, a local
  accumulator per pack can merge many tiny row contributions *before* a
  shared `StateTable` update; that is the stronger candidate.
- **Row broadcast:** `BroadcastStreamingOOCPrimitive` currently puts each
  large-side tile and summary into state tables, allocates a match, and
  submits work per tile. A pack-aware variant could acquire an aligned run
  of matrix tiles once and apply the corresponding summary members in one
  admitted task. The output still needs a grouped handoff or it retains
  per-tile puts.
- **Aligned join:** `JoinStreamingOOCPrimitive` matches exact tile keys.
  Soft pack ranges could prune or batch candidate keys, but cannot replace
  exact lookup because two streams need not have identical pack boundaries.
  The isolated join showed the smallest blocksize penalty here; it is not
  the first primitive to specialize.
- **Sparse matrix–vector:** the existing specialized/partitioned paths already
  address some sparse accumulation cases. This follow-up did **not** run a
  sparse end-to-end A/B, so it cannot assign a speedup to arbitrary sparse
  packs. The PageRank-style path is not interchangeable with general sparse
  matmul or join.

The source is already row-major, but its records and cache entries are still
individual tiles. A useful integrated prototype should preserve exact
tile→pack-member lookup, reserve/pin the **whole physical pack before I/O**,
and first target a single scan-heavy reduction such as `t(W) %*% X`. It
should compare three versions at the *same logical block size and values*:
baseline, physically packed storage with existing tile callbacks, and packed
storage plus pack-aware local accumulation. Record source and spill bytes,
pack reloads, get/put counts, StateTable updates, task count, and wall time.
That experiment would separate physical packing from primitive batching.

The older global `OOCPackedCache` route was not used in these dense controls
and previously failed a small GNMF correctness run. The Twitter source-pack
experiment below is a separate, opt-in sparse path added afterward.

## Twitter-2010 source-pack follow-up (2026-09-20)

The opt-in sparse source reader now groups consecutive **present** 10k tiles
by decoded memory charge, with a 16 MiB cap. A source group becomes one
`PackedBlock` cache entry using the source group's existing reservation;
there is no global packed-cache wrapper or timer-based regrouping. The
specialized partitioned sparse matvec consumes these packs. The native graph
is unchanged (41,652,230 vertices, 1,468,365,182 edges, 17,359,722 present
tiles, 21 GiB on disk). This groups decoded source records; SequenceFile
still decodes records individually.

Both arms used the same rebuilt SystemDS JAR, 10k native input, 16 GiB cgroup,
12 GiB JVM, 16 visible processors, direct I/O, cold input, and identical
PageRank DML. Only `sysds.ooc.materialized.partition.bytes` differed:

| Iterations | Path | Wall | cgroup read | cgroup write | Cache puts | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Unpacked | 143.95 s | 116.54 GB | 0.07 GB | 17,380,552 | complete |
| 1 | Source packs, 16 MiB | 51.98 s | 52.06 GB | 18.22 GB | 30,214 | complete |
| 3 | Unpacked | >300 s | 239.21 GB at timeout | 0.29 GB | n/a | timeout |
| 3 | Source packs, 16 MiB | 65.34 s | 89.88 GB | 18.94 GB | 63,537 | complete |

The completed one-iteration outputs agreed to 8.6e-17 maximum absolute
difference; both rank vectors summed to 1.0000000000005986. The forced-spill
component test passed with deliberately missing sparse tiles, and the
three-iteration packed run completed without OOM. The timeout is not a
completed runtime or numerical comparison. These are one cold repetition per
arm, not stable performance estimates. Results and telemetry are under
`/workspace/data_dir/twitter-small-tiles-diagnostic/source-{pack,unpacked}-page-rank-{1,3}iter/`.
