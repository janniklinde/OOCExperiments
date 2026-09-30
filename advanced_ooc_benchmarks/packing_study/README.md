# Source-layout and reactive packing study

The [native packed KMeans chain](PACKED_KMEANS_CHAIN.md) implements source-to-contraction
pack preservation and compares 200M x 2 and 1B x 2 under a 4 GiB cgroup. It records
the cache-call reduction, mixed runtime results, remaining spill/read costs, and
admission/lifetime fixes rather than attributing all gains to packing.

The [range-contract validation](range-contract-findings.md) compares the actual
rectangle filter with exact partition discovery using identical bounded
build/probe consumers, including KMeans and GNMF operator chains.

The [byte-only and bucket-capacity follow-up](bucket-capacity-findings.md)
removes the experimental 32-tile closure constraint and varies matching
bucket size, cache capacity and work allowance under enforced cgroups.

The [packing-policy study](policy-findings.md) compares arrival, staged,
coordinated bucket and sorted layouts across shapes and reuse horizons, including
direct SSD reads. See its [methodology](policy-study.md) for the distinction
between the original forced-writer-flush measurements and the subsequent
direct-read/buffered-write validation.

The [join scheduling follow-up](join-expansion.md) holds physical packs fixed
and compares one-hop tasks, linear batching and bounded overlap expansion.

The broader algorithm-kernel and balanced-join follow-up is documented in
[broader-findings.md](broader-findings.md), with setup and reproduction in
[broader-study.md](broader-study.md). It uses actual SystemDS cache/store APIs,
unlike the original isolated simulator described below.

The bounded-buffer 1D packing follow-up is documented in
[staging-findings.md](staging-findings.md), with implementation and reproduction
details in [staging-study.md](staging-study.md).

This is an isolated prototype, **not** a SystemDS cache implementation. It tests
the physical design proposed for skinny/ultra-small tiles before changing the
`MaterializedStore` contract:

- A strict row-major source writer groups consecutive logical tiles up to a
  byte target. Except for the final partition, groups are approximately equal
  sized. This grouping is known at source-preparation time.
- A reactive spill writer groups nearby *available* tiles from one stream when
  hot bytes exceed a bound. Its groups are determined by arrival and eviction
  order, not by a fixed logical partition mapping. A `maxGap` bound limits how
  far it reaches in logical indices; it is a heuristic, not a semantic promise.
- Every logical index has an exact reference to `(physical pack, slot)`. The
  physical pack has one metadata record, sorted contained indices, offsets,
  and a conservative `[min,max]` hint. The hint can prune candidates without
  asserting that a tile *must* be in a particular pack. Any candidate still
  requires exact index lookup. A many-to-one reference scheme could replace
  multiple `BlockEntry` objects in a real cache.
- A pack-aware operator schedules one task per X pack; a tile-mode operator
  schedules one task per logical X tile. Both use the same FP64 values. Tested
  kernels are a scan/reduction, a row-aligned broadcast join, and skinny
  matrix multiplication. Checksums must agree.
- Cold full and stride-10 reads use an 8 MiB resident-pack cache. Each pass
  first evicts the named file from Linux page cache. The result records both
  application read calls/bytes and cgroup-v2 `io.stat` physical `rbytes`.

The case matrix covers 8 KB and 80 KB logical tiles over a 2M-by-10 dense
matrix (160 MB payload), plus 80 KB tiles over 20M-by-10 (1.6 GB payload).
An opt-in `--geometry huge --read-only` case uses 100M-by-10 (8 GB payload)
under the same 6 GiB cgroup; warm operators are skipped to avoid retaining
the entire source in memory.
The source is always row-major. Reactive arrivals are row-major, shuffled
within 64-tile windows, or globally shuffled with a fixed seed. The default
physical target is 512 KiB; other cases compare one tile per pack and 2 MiB.

Run in the Devcon benchmark container, where `/sys/fs/cgroup/devcon` supports
the memory and I/O controllers, from the repository root:

```bash
/opt/devcon/env/python/bin/python advanced_ooc_benchmarks/packing_study/run_study.py \
  --results /workspace/data_dir/packing-study/my-invocation \
  --geometry all --repetitions 5 --read-repetitions 3
/opt/devcon/env/python/bin/python advanced_ooc_benchmarks/packing_study/summarize.py \
  /workspace/data_dir/packing-study/my-invocation
```

`--case small8-reactive-global --hot-mib 16 --gap 128` isolates one
fill/locality setting. The runner removes its own binary pack files after
each case; logs, cgroup metrics, and telemetry remain. `--keep-packs` retains
the binaries for inspection. It never touches benchmark datasets.

Important limitations: the files hold raw doubles rather than SystemDS
SequenceFile/MatrixBlock serialization; warm kernels load all packs into RAM
and do not materialize output matrices; the prototype does not implement
SystemDS allowance, ownership, purging, readahead, or spill reload. Thus the
study separates physical I/O and per-tile scheduling costs, but its operator
times are **not** predictions for a SystemDS end-to-end job. See
[FINDINGS.md](FINDINGS.md) for measured outcomes and integration guidance.
The [real OOC follow-up](REAL_OOC_RESULTS.md) compares actual SystemDS
scan, matmul, broadcast, join, KMeans, and GNMF runs at thin logical tiles;
it explicitly separates these blocksize controls from the isolated pack
prototype. It also includes the opt-in Twitter 10k source-pack PageRank
result. `run_twitter_source_probe.py` reproduces the cold, cgroup-limited
comparison from an already prepared native graph.
