# Non-aligned packing: locality, false candidates, and cache reloads

## Setup

Same component workload as atomic-packing.md: X is 100M x 2 FP64 (1.6 GB),
Y = X P is 100M x 8 (6.4 GB), followed by P = X^T Y and normalization.
Blocksize 1000, maximum 32 logical tiles / 512 KiB per pack, eight contraction
lanes, 4 MiB atomic reservation cap, four compute threads plus I/O threads,
3 GiB heap. Cache is 64 MiB except the explicit 256 MiB control.
Each JVM warms up for 30 seconds and runs three measured iterations. Numerical
results are checked against an independently computed Gram matrix every time.

These are full matrix operations, not random point lookups. Except the ideal
control, four source producers claim 17-tile chunks and submit each chunk in
reverse order. Packing is lane-local and concurrent. A locality key closes a
pack on entering another row window; it does not guarantee matching partitions
between inputs. Y is packed live by the projection's four workers.

- Aligned: prebuilt contiguous X packs; every Y pack lies within one X pack.
- Window 32: locality key floor(tileRow / 32), or 32,000-row bands.
- Window 128: 128,000-row bands, allowing more gaps within each pack.
- Arrival order: no locality key changes; fill packs from each producer's
  arrivals. This still has chunk locality; it is not globally randomized input.

Packing concurrency makes the precise layout slightly nondeterministic between
JVMs. Cache-size controls therefore have very similar, not identical, pack
membership. All runs used buffered I/O without a page-cache reset or cgroup
isolation. No production primitive, cache policy, or planner behavior changed
in this study; only the component harness gained diagnostics.

## Results

Contraction-only medians over three iterations; decimal GB. The table uses the
latest membership-checked runs where available; earlier repeats are also saved.

| Layout | Cache | Elapsed (s) | CPU (s) | X reload (GB) | Y reload (GB) | Total reload (GB) | Total / 8 GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aligned | 64 MiB | 1.658 | 14.45 | 1.588 | 6.391 | 7.977 | 1.00x |
| Window 32 | 64 MiB | 2.126 | 16.93 | 1.588 | 6.483 | 8.071 | 1.01x |
| Window 128 | 64 MiB | 3.170 | 21.06 | 1.587 | 9.932 | 11.519 | 1.44x |
| Arrival order | 64 MiB | 3.217 | 24.83 | 1.587 | 15.016 | 16.603 | 2.08x |
| Arrival order | 256 MiB | 2.551 | 17.25 | 1.540 | 8.672 | 10.212 | 1.28x |

X remains approximately a single scan in every case. Amplification is on Y.
The slightly sub-payload totals reflect retained cache contents and differences
between FP64 payload and serialized accounting. Independently computed medians
need not sum exactly.

The earlier 64 MiB arrival-order run measured 3.310 s / 16.667 GB; the earlier
window-128 run measured 3.088 s / 11.538 GB. Thus the main read-volume trends
repeated across JVMs. These small samples are not precise statistical runtime
claims. Projection times fluctuate substantially, so whole-iteration runtime
is not used to attribute contraction overhead.

Every timed projection and contraction reported zero process storage-read
bytes. These cache reloads came from the OS page cache, not the physical SSD.
The results measure the combined cache reload/deserialization/compute path.

## Exact cause: bounding boxes select unrelated packs

Before timing contraction, the harness counts metadata candidates and sums their
requested Y sizes. After the final timed iteration, a separate inspection scans
both inputs and maps each tile row to its owning X pack. It then counts how many
X packs genuinely share a tile with each Y pack. These inspection reads are
outside all reported timings and I/O deltas.

Exact membership results for the final iteration:

| Layout | Y packs | Bounding-box candidate pairs | Pairs sharing tiles | Pairs sharing no tiles |
|---|---:|---:|---:|---:|
| Aligned | 12,500 | 12,500 | 12,500 | 0 |
| Window 128 | 14,350 | 40,750 | 16,312 | 24,438 (60.0%) |
| Arrival order | 12,502 | 65,806 | 12,502 | 53,304 (81.0%) |

This refines the earlier hypothesis that Y simply had to serve many genuinely
overlapping X packs. In the arrival-order case, each Y pack has exactly one
real X counterpart. But an X pack contains separated row runs, and its bounding
box covers the gaps too. Other Y packs fall inside these gaps. The current
filter returns them; the operator loads them and only its tile-level merge
discovers that there is no work to do.

For that arrival-order iteration, candidate requests total 33.817 GB of Y
in-memory pack charges, including 27.393 GB for wholly irrelevant pairs.
Actual reload is smaller because repeated acquisitions can hit cache or share
an in-flight load. This requested-byte metric is not physical I/O, nor exactly
the serialized-byte measure in the results table.

Window 128 has both effects: false candidates dominate the candidate count,
but 16,312 real pairs for 14,350 Y packs also demonstrate genuine cross-pack
matches. Better membership filtering cannot eliminate that genuine reuse.

Window 32 needs only about 17,878 candidate pairs and 6.920 GB of requested Y
pack charges. Its nearly single-scan reload shows that strict alignment is not
necessary. It creates more, smaller packs (8,824 X packs versus 3,125 aligned),
so its runtime overhead cannot be attributed to read amplification alone.

Increasing arrival-order cache to 256 MiB keeps the same approximate candidate
count (~65,829) but reduces reload substantially. This demonstrates that cache
retention matters too; more cache does not remove false candidate processing.
No per-eviction trace was collected, so these results do not identify an optimal
replacement policy.

## Recommendation

Before introducing stricter partitioning, add an optional compact membership
summary to descriptors, such as a few occupied row intervals. Keep the existing
bounding-box index as the coarse filter, then reject definitely disjoint packs
using those summaries before acquiring cache leases. A conservative summary
must never reject a genuine match. It need not expose a fixed partition mapping
or force strict partition boundaries.

This is particularly promising for these chunk-local packs: they contain gaps
between relatively few contiguous runs. For arbitrary scattered data, summaries
may need to be capped and become conservative again. PageRank streaming need
not pay for this metadata.

After removing false candidates, study bounded grouping of neighboring left
packs or cache reuse hints for genuinely shared Y packs. Retaining X already
solves its repeated reads; increasing the atomic reservation alone does not
make the coarse lookup more selective. No such filtering optimization was
implemented or benchmarked in this study.

## Reproduction and artifacts

```bash
cd /workspace/systemds
/opt/devcon/env/bin/mvn -q -Dtest=PackingMaterializerTest test
cd /workspace/OOCExperiments
JAVA_BIN=/opt/devcon/env/java/current/bin/java \
  bash advanced_ooc_benchmarks/packing_study/run_concurrent_packing.sh \
  100000000 67108864 32 128 8 4194304
```

The fourth argument selects -1 (aligned), 32, 128, or 0 (arrival order).
The second selects cache bytes; use 268435456 for the larger cache control.
Seven focused component tests passed. All seven large JVM runs completed and
passed numerical validation; three also completed exact membership inspection.

Raw logs: data_dir/packing-locality-100m-{w32,w128,unordered64,unordered256,
unordered-exact,w128-exact,aligned}.log. All measured phase records and geometry
counters are in nonaligned-packing-results.csv. Exact membership totals remain
in the raw logs and the table above. Spill files are temporary and removed by
the fixture; no additional persistent large datasets were created.
