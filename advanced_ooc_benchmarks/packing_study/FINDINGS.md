# Preliminary packing study — 2026-09-19

## Question and scope

Can physical groups of adjacent, small tiles reduce OOC cache-entry and
operator overhead while preserving exact per-tile access? Source reads must
be assessed separately: unlike reactive spill, the source writer can choose
a strict row-major layout with near-equal byte-sized partitions.

This study uses an isolated Java prototype (`ReactivePackExperiment.java`) and
cgroup-v2 physical I/O measurements. It is deliberately **not** an integrated
SystemDS benchmark. The raw case logs and telemetry are under
`/workspace/data_dir/packing-study/20260919-{small-cold-01,large-cold-01,source-reps-01,huge-source-01,huge-source-02,global-h*-g*}`.
All named cases completed. In cases with warm operators, logical checksums
agreed between tile- and pack-scheduled modes.

## Design tested

The source layout writes consecutive row-major tile indices into a pack up
to a byte cap. With 80 KB tiles and a 512 KiB cap, most packs hold six tiles
(480 KB); with 8 KB tiles, most hold 65 tiles (520 KB). A strict cap is
enforced; the last pack can be short.

Reactive packing keeps a bounded set of hot tiles and, under pressure, starts
at the oldest tile and collects nearby *currently available* indices into a
pack. It does not defer a required eviction indefinitely to await absent
neighbors. Thus it has a strict byte upper bound but cannot guarantee a
near-target lower bound for adversarial arrival orders with finite staging
memory and a narrow neighborhood. Both paths maintain an exact per-tile
reference; pack `[min,max]` ranges are only conservative join-pruning hints.
This does **not** make a logical index belong to a fixed partition.

## Physical source reads

The repeated cold-read comparison uses 6 GiB cgroup memory, three file-cache
evictions/read passes, an 8 MiB resident-pack cache, and cgroup `io.stat`
`rbytes`. Times below are the median of three passes; MB are decimal.

| X payload / tile | Layout | Packs | Full-scan physical read / time | Every 10th tile physical read / time |
|---|---:|---:|---:|---:|
| 1.6 GB / 80 KB | single tile | 20,000 | 1.60 GB / 533 ms | 168 MB / 268 ms |
| 1.6 GB / 80 KB | row-major 512 KiB | 3,334 | 1.60 GB / 550 ms | 1,224 MB / 465 ms |
| 160 MB / 8 KB | single tile | 20,000 | 160 MB / 98 ms | 24 MB / 123 ms |
| 160 MB / 8 KB | row-major 512 KiB | 308 | 160 MB / 91 ms | 160 MB / 88 ms |

For the 1.6 GB / 80 KB case, packing cuts full-scan read *calls* by about 6×
but neither bytes nor elapsed time improve on this SSD; selective reads get much worse
because a requested tile pulls neighboring unneeded tiles, plus filesystem
readahead. The 8 KB case shows that very small reads can become call-bound:
packing makes even the every-10th-tile pass faster here, despite reading far
more bytes. This is a crossover, not a universal source-packing win. The
intermediate 160 MB / 80 KB cases also showed similar full-scan times and
strong selective-read overfetch at 512 KiB and 2 MiB. The source layout must
therefore be chosen for expected access pattern, not just tile count.

A separate 8 GB source (100M-by-10, 100,000 tiles) exceeds the 6 GiB cgroup
limit. This was read-only: no attempt was made to hold the whole source in
the Java heap. With three cold passes, the median full scan was **3.42 s**
for one tile/pack versus **2.86 s** for 512 KiB row-major packs; both read
8.00 GB physically. The one-in-ten-tile pass was **1.73 s / 0.84 GB** versus
**2.44 s / 6.12 GB**. An independent two-pass invocation showed the same
direction (full: 3.66 versus 3.20 s; selective: 1.49 versus 3.17 s).
This strengthens the result that source packing *can* help a cold sequential
scan once the data exceed RAM, while the selective-read penalty remains large.
These are raw packed files using buffered I/O; SystemDS source uses
SequenceFile records and can use direct I/O, so the exact crossover must be
remeasured there.

## Reactive fill and soft locality

All rows below are a 160 MB, 2M-by-10 matrix with 20,000 logical 8 KB tiles,
512 KiB pack cap, and globally shuffled arrival. `Underfilled` counts packs
below 75% of cap. Candidate/actual counts refer to pairs of X and aligned
broadcast-vector packs; candidate pairs use only the range hints.

| Hot staging | Max index gap | Packs | Mean payload | Underfilled | Full-scan pack reads | Candidate / actual pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 8 MiB | 32 | 857 | 187 KB | 757 | 857 | 1,863 / 1,153 |
| 16 MiB | 32 | 433 | 370 KB | 160 | 433 | 620 / 620 |
| 32 MiB | 32 | 375 | 427 KB | 90 | 375 | 480 / 480 |
| 8 MiB | 128 | 337 | 475 KB | 46 | 340 | 930 / 669 |
| 32 MiB | 128 | 335 | 478 KB | 36 | 335 | 455 / 455 |
| 8 MiB | 512 | 314 | 510 KB | 10 | 416 | 955 / 707 |
| 8 MiB | 2,048 | 311 | 514 KB | 4 | 534 | 1,002 / 743 |

Row-major reactive arrival formed 308 packs, essentially all near target;
64-tile-window shuffling also formed 308. Global shuffle with 8 MiB staging
and a narrow gap did not. Enlarging the gap repairs fill with little staging,
but can create wider, holey index ranges and extra join candidates. At gaps
512 and 2,048, a logical full scan even reloads packs in the bounded
application cache: more pack reads than physical packs. The Linux page cache
kept these small-case rereads from becoming extra device bytes here; a larger
working set could make them physical rereads. Larger staging repairs fill
while keeping a narrower range, at a memory cost. These figures quantify why
the range is a *soft hint*, not an ownership rule.

## Operator scheduling

The prototype's warm 8-thread kernels hold packed input bytes in RAM, compare
task-per-logical-tile to task-per-pack, and checksum the same calculations.
Representative medians (five warm repetitions in the first sweep; ms):

| Matrix / tile | Operator | Tile tasks | Pack tasks | Time, tile → pack |
|---|---|---:|---:|---:|
| 160 MB / 8 KB | scan/reduce | 20,000 | 308 | 16.3 → 4.6 ms |
| 160 MB / 8 KB | row-broadcast join | 20,000 | 308 | 14.9 → 4.1 ms |
| 160 MB / 8 KB | skinny matmul | 20,000 | 308 | 18.6 → 13.5 ms |
| 1.6 GB / 80 KB | scan/reduce | 20,000 | 3,334 | 36.1 → 32.3 ms |
| 1.6 GB / 80 KB | row-broadcast join | 20,000 | 3,334 | 40.5 → 37.6 ms |
| 1.6 GB / 80 KB | skinny matmul | 20,000 | 3,334 | 117.8 → 112.9 ms |

The 8 KB operator case has a strong scheduling crossover; the 80 KB case
shows only modest benefit, especially in compute-dominated matmul. Warm times
vary between sweeps/JIT states, so these are directional microbenchmark
results rather than paper-ready speedups. No conclusion about a real
SystemDS `MatrixBlock` operator, output materialization, or cache lock costs
follows directly from these kernels. The prototype also retains a Java
`Ref` object per logical tile, so it does not measure the anticipated memory
saving from replacing multiple real `BlockEntry` objects with one shared
physical entry and compact integer references. The integrated experiment should compare
the same operators with a source scan and a forced spill/reload at 8 KB,
80 KB, and at least one dense full-sized block.

## Prior art and interpretation

The range hint plus exact member lookup resembles SST-file min/max key
filtering in [RocksDB's file indexing](https://github.com/facebook/rocksdb/wiki/Indexing-SST-Files-for-Better-Lookup-Performance), not a deterministic
logical partition map. [GraphChi](https://www.cs.cmu.edu/~guyb/papers/KBG12.pdf)
uses vertex intervals and disk shards to create predictable sequential
access. [X-Stream](https://www.sigops.org/s/conferences/sosp/2013/papers/p472-roy.pdf)
motivates sizing streaming partitions to amortize I/O while keeping them
within fast-memory capacity. [Morsel-Driven Parallelism](https://faculty.cc.gatech.edu/~jarulraj/courses/4420-s19/papers/18-execution/p743-leis.pdf)
supports bounded units of operator work rather than one task per tiny item.
These are design analogies, not evidence that their implementations can be
transplanted directly into SystemDS.

## Recommended integration gate

1. Keep source layout separate from eviction packing. Default to existing
   source format until a row-major pack layout can be benchmarked against
   realistic full, selective, and repeated source scans. A source pack should
   carry exact tile offsets and preserve normal source-stream semantics.
2. Prototype a reactive physical pack as one `BlockEntry`/spill handle with
   compact per-tile references, bounded payload, and no partition ownership
   contract. Pack only eligible cold, immutable neighbors in the same stream.
   Keep hot mutable accumulators ungrouped. A max-gap or range-width budget is
   needed so fills do not destroy join selectivity.
3. Add an optional pack-aware batch callback to a few streaming primitives;
   preserve existing tile callbacks as the default. Allowance must cover a
   whole physical reload and the operator output. This is where the 8 KB
   scheduling benefit should be tested in real OOC execution.
4. Measure separately: physical source bytes/read calls; spill bytes/read
   calls; cache metadata bytes and reloads; pack fill distribution; candidate
   versus actual join pairs; task count; wall and CPU time. Reject the change
   if full scans do not benefit and selective reads or spill reloads regress
   significantly under the intended workloads.

No production cache change is justified yet. The strongest current result is
that source packs are easy to fill with row-major layout but can over-read
badly, while reactive packs need an explicit locality/fill trade-off and
pack-aware consumers to turn fewer metadata entries into actual speedups.
