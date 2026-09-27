# Findings: bounded staging and 1D packing

## Bottom line

The lightweight sorting heuristic works: it produces full packs with much
tighter row ranges, without the previous close-on-window-change fragmentation.
The sorting/selection itself is cheap. However, minimizing row range is not
equivalent to minimizing matched-input reloads. Sorting a newly produced factor
can destroy the useful affinity to the input pack that produced it.

Implementation, budgets, reproduction, and limitations are in
[staging-study.md](staging-study.md). All measured cases, including preparation,
are in [staging-results.md](staging-results.md); the adjacent CSVs contain phase
measurements and pack/partner/stream-I/O diagnostics.

Final validation: **96 completed cases, 480 measured iterations**, matching
checksums across policies, plus two focused JUnit methods covering **62 numerical
forced-spill configurations**. The final sweep had no observed OOMs or stalls.
One prototype source-executor deadlock was fixed before these sweeps, as recorded
in the implementation notes. No production classes or benchmark plans were
modified for this experiment, and temporary spill files were cleaned up.

## 1. Fullness problem solved; no consistent single-input speedup

Primary sweep, globally shuffled arrivals, 16 MiB cache, median seconds over
five iterations:

| Chain | Arrival | Old row-window | 50 MiB sorted staging |
|---|---:|---:|---:|
| LMCG normal operator, 32M×2 | 0.181 | 1.589 | 0.190 |
| KMeans, 8M×2, k=32 | 0.440 | 0.816 | 0.433 |
| GNMF, 8M×2, rank16 | 0.841 | 1.965 | 0.957 |
| GNMF, 1M×128, rank16 | 0.679 | 0.661 | 0.713 |

For random 32M×2 LMCG, the old window policy produced 31,970 groups, arrival
1,002, and staging 1,012. All packed variants reloaded about 0.502 GB per
iteration. Staging avoids almost all fragmentation, but does not reduce the
work of a full single-input scan. Repeated measurements with another seed
show the small LMCG/KMeans differences changing direction; they do not support
a reliable sorting speedup. The 8M×1 resident control similarly shows only
millisecond differences between the full-pack policies, with zero steady-state
reloads; old windows still create about 8,000 groups instead of 252.

## 2. Locality really improves, but the locality metric misses producer affinity

On globally shuffled GNMF 8M×2:

- X's average pack row span shrinks from **7,463 to 293 logical tile rows**,
  with essentially the same number of packs (253 versus 252).
- Initial W's mean span shrinks from about **4,817 to 146**.
- Nevertheless, after one W update, arrival packing has **zero** W packs
  spanning multiple X packs; staged W has **391 of 2,007** crossing X packs.
- Median reload increases from **2.291 to 2.652 GB**, while writes remain
  approximately **1.022 GB**. Repeats preserve this amplification.

Arrival-order W follows the tile set of the X task producing it, even when
that tile set is scattered in logical row space. Sorting W by logical row
interleaves members of different X packs. A later X-driven task must acquire
more W packs; unrelated members are loaded and can be reloaded by later tasks.
Thus a tighter bounding box can coexist with worse physical match affinity.

This is not evidence that all spatial clustering fails: staging is sharded
across four producer lanes, not one global sorter. Also, the join uses an exact
membership index. It does not benefit from reduced false positives in a
bounding-box-only index, because it already avoids those false positives.

## 3. Instrumentation and controls identify the extra traffic

GNMF 2M×2/rank16, shuffled inputs, median **iterations 2–5**, MB in decimal:

| Policy | X reload | W reload |
|---|---:|---:|
| Arrival | 58.75 | 500.62 |
| Sort X and every W update | 58.49 | 615.47 |
| Sort X once; leave W in arrival order | 58.75 | 501.13 |

The extra traffic is the repeatedly matched factor, not rereading more X or
writing substantially more W. In the same case, total median per-iteration
reload rises from 0.560 GB (arrival) to 0.562/0.606/0.675 GB for 4/16/50 MiB
staging. Bigger sorting windows can mix more producer groups, not just find
better groups. At a 4 MiB cache, the 1M×2 case grows from 0.284 to 0.392 GB
reload with 50 MiB staging.

Sorting/selection for updated W in the 8M×2 cases takes a median **1.43 ms of
summed lane time**, maximum 1.97 ms in the sampled main/repeat/follow-up runs.
Initial X selection has median 0.79 ms. This excludes publication, array
compaction, and ownership transfer; it is not a complete packing CPU profile.
Nevertheless, the regressions cannot be explained by these sorting intervals
alone. Equally buffered FIFO controls preserve the low reload volume, further
separating buffering overhead from the spatial reordering effect.

## 4. Initial sorting can help the first join, but repeatedly sorting outputs is different

GNMF 8M×2/rank16, shuffled inputs, follow-up seed73:

| Policy | First-iteration reload GB | Five iterations s | Including preparation s |
|---|---:|---:|---:|
| Arrival throughout | 8.063 | 4.757 | 5.085 |
| Sort initial inputs and each new W | 7.064 | 5.189 | 5.554 |
| Sort initial inputs only | 7.083 | 4.718 | 5.077 |

The initial independently shuffled inputs benefit from clustering. After the
first update, retaining producer affinity is better. Initial-only sorting
removes the repeated penalty, but its total time here is effectively equal to
arrival packing; this does not justify mandatory source reorganization.

## 5. The consuming primitive can eliminate the penalty

GNMF 1M×128/rank16, shuffled inputs, same follow-up cohort:

| Task driver / packing | Median iteration s | Reload GB |
|---|---:|---:|
| X-driven / arrival | 0.687 | 2.567 |
| X-driven / sorted staging | 0.690 | 3.000 |
| W-driven / arrival | 0.543 | 2.286 |
| W-driven / sorted staging | 0.544 | 2.285 |

Each X pack contains one tile; W packs contain several. Driving from W gathers
the matching X packs into one admitted task and uses every W member before
releasing the lease. That removes the repeated partial-use penalty. No strong
matching-partition assumption or repartitioning is needed. This control also
shows that the staged packs themselves are not intrinsically slow.

## 6. Larger-data checks and an important timing limitation

The larger skinny control uses **256M×1** X (2.048 GB), about 256,000 logical
8-KB tiles. Arrival and staged packing reduce that to 8,002 and 8,012 packs.
Both reload **2.046 GB** per iteration and write no intermediates. Median
iteration times were 0.759 and 0.722 s, while staging preparation was slower
(0.755 versus 0.655 s). This is another near-parity full-scan result, not evidence
of reduced read volume from sorting.

The expanded GNMF control uses **8M×2/rank64**, with a **4.096 GB W**, larger
than the 3 GiB heap. Each W tile already nearly fills a pack, so staging cannot
reduce its entry count. Both policies consistently reload about **8.438 GB**
and write **4.094 GB** per iteration, with roughly 30–32 CPU seconds.

Its wall times were not stationary: in the first larger-data cohort arrival
and staging medians were 4.483 and 7.014 s. Reversing/repeating their order also
produced slow arrival runs, including a 12.251 s arrival median in the final
seed91 pair. There were episodes of substantial host I/O pressure, changing
physical-read counts, and slow iterations even with zero process-accounted
storage reads. GC logging showed short individual pauses, not multi-second
stop-the-world pauses; the sampled benchmark process had zero VmSwap.

These observations rule out explaining this as increased serialized read
volume or direct sorting CPU cost. They do **not** establish the exact source
of the additional I/O/publication waiting. The write-heavy largest case needs
controlled I/O isolation and phase tracing before assigning a causal wall-time
penalty to staging. All runs remain in the results rather than discarding the
slow observations. The smaller, reproducible rank16 W-reload amplification is
a separate finding supported by per-stream counters and membership changes.

## Recommended next integration step

Keep the staging adapter optional, with a configurable 1D grouping key. Logical
row index is useful for independently arriving source tiles, but a producing
task's input-pack/band affinity is often the better key for intermediates.
Avoid sorting a layout that already matches its consumer. Let matched-input
primitives choose a driving side and retain a pack while processing its bounded
partner set. Fall back to arrival packing for full scans or when no useful
affinity is known.

Do not infer that this is ready to enable globally: these are manually assembled
operator chains with real SystemDS cache spilling, not full benchmark DML plans.
Cache sizes are not total process memory limits. Most reloads are OS-cache-backed;
there is no cold-SSD throughput claim. Sparse/multi-column-tile shapes and demand-
driven early flushing of live dependencies remain separate validation work.
