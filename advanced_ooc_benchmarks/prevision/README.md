# PreVision pilot baseline

This optional pilot compares [PreVision's published engine](https://github.com/snu-dbs/prevision)
with the suite on two dense, FP64 workloads at the 16 GiB cgroup profile:

| Case | Computation | Input | Other arms |
| --- | --- | --- | --- |
| `gnmf_prevision` | Three rank-16 multiplicative GNMF updates, then both factors | Nonnegative `gnmf_d32`, 4,000,000 × 1,000 | SystemDS OOC/Spark, Dask |
| `gram_prevision` | `t(X) %*% X`, materialize the Gram matrix | `dense_d32`, 4,000,000 × 1,000 | SystemDS OOC/Spark, NumPy, Dask |

The GNMF factors use the suite's deterministic FP64 initialization. PreVision executes
the same update order (H, then W) and epsilon, and the adapter exports complete FP64
outputs inside the timed cgroup. The Gram arm is an algebraic building block, not a
separate learning algorithm. No PreVision code is vendored here: `setup.sh` fetches
commit `e4c4e96cd2e884b6f0d97e6b3410623fa4d13f59` and links this suite's
adapter to it. Consult the upstream repository's license before using its code.

## Code layout

The raw learning algorithm is [gnmf/prevision.c](../gnmf/prevision.c), beside its
SystemDS, NumPy, and Dask counterparts. It only constructs the lazy GNMF DAG;
it contains no dataset preparation, command-line parsing, cache configuration,
or output I/O. [gram/prevision.c](../gram/prevision.c) follows the same pattern.

The reusable support stays in this directory:

- `workloads.h`: the small interface between raw workload DAGs and the adapter.
- `adapter.c`: bounded FP64 import, input opening, execution/materialization,
  factor export, and the native command-line entrypoint.
- `setup.sh`: builds the pinned upstream engine and links the workload sources.
- `prepare.py`: geometry-qualified preparation and provenance/readiness checks.
- `run.py`: cache/thread configuration, process isolation, bounded logs, and cleanup.
- `memfd_shm.c`: the optional container shared-memory compatibility shim.
- `make_plan.py`: selects the opt-in pilot workloads from the main plan.

To add another workload, put its DAG builder in that workload's directory, declare
its entrypoint in `workloads.h`, and link it in `setup.sh`. Reuse the import/export
and launcher infrastructure, extending their operand/output dispatch as needed.
After changing a C source, rerun `prevision/setup.sh <prefix> [existing-source]`.
An algorithm-only adapter rebuild does not invalidate prepared data: the adapter
hash remains provenance, while generator version, source fingerprint, shape, tiling,
and initialization parameters determine input compatibility. Changes to import or
initialization semantics must bump the generator version.

PreVision needs a POSIX shared-memory arena. Its 10 GiB data cache plus two 128 MiB
metadata stores fit under the 16 GiB cgroup, with remaining room for computation.
The container's `/dev/shm` must be at least 11 GiB; 12 GiB allows some slack. Run
only one PreVision process at a time: the published BufferTile uses fixed shared-memory
names. The wrapper serializes this suite's PreVision runs and cleans those names after
killed runs, but cannot coordinate with an unrelated PreVision process.
On Linux containers where `/dev/shm` cannot be enlarged, set
`defaults.resources.prevision_arena: memfd` in a host-specific plan. The optional
shim replaces only BufferTile's named shared-memory opens with Linux `memfd`
objects; the same memory-backed pages remain charged to the benchmark cgroup.
The default `posix` mode uses the unmodified upstream shared-memory path.

`defaults.resources.prevision_blas_threads` and `prevision_elementwise_threads`
default to four and one respectively, independently of the host task-thread
allowance. They can be tuned independently without modifying the upstream engine. The published
elementwise kernels create/join a fresh pthread team for each tile, so using all
CPUs for small factor tiles can cost more than it saves. For a local pilot, e.g.:

```sh
python prevision/make_plan.py --arena memfd --threads 22 \
    --blas-threads 4 --elementwise-threads 1 --tile-rows 40000 --tile-cols 1000
```

The host allowance still applies to the other arms; Dask distributes it across
worker processes, rounded down to an equal thread count per worker.

On a Docker benchmark host:

1. Set `SHM_SIZE=12gb` in `advanced_ooc_benchmarks/docker/.env`, then rebuild and
   recreate the container (`./bench.sh build`, `./bench.sh down`, `./bench.sh up`
   from the `docker/` directory). The updated image has the C compiler and
   OpenBLAS headers needed for this optional baseline.
2. In `./bench.sh shell`, run
   `/workspace/advanced_ooc_benchmarks/prevision/setup.sh /bench/data/tools/prevision`.
   This requires network access to GitHub only for the initial source fetch.
3. Still in that shell, run
   `/opt/bench-venv/bin/python /workspace/advanced_ooc_benchmarks/prevision/make_plan.py`.
   This creates `benchmark-plan-prevision.generated.yaml` beside the main plan,
   enabling *only* the two pilot workloads. The normal sweep remains unchanged.
4. From the `docker/` directory on the host, run
   `./bench.sh run /workspace/advanced_ooc_benchmarks/benchmark-plan-prevision.generated.yaml`.
   Use `--only 'gram_prevision*'` or `--implementation prevision-gram` after the
   plan path to narrow the run.

The converter reads the suite's existing canonical row-major `X.f64` and writes
PreVision TileStore tiles directly, one rectangular FP64 tile at a time, before
timing. The default 32 GB pilot uses 40,000 × 1,000 tiles: 320 MB per input tile
and 100 tiles, matching the published experiment's tile bytes and count without
changing our matrix shape. Set `parameters.prevision_gnmf_tile` (rows) and
`prevision_gnmf_tile_cols` (columns) for GNMF; Gram has the analogous
`prevision_gram_tile` and `prevision_gram_tile_cols` parameters. These extents are
independent of SystemDS blocks and Dask chunks. Preparation accepts `--tile-rows`
and `--tile-cols`; the old square `--tile` remains compatible with old manifests.
Variants are stored in geometry-qualified `gnmf-tr<rows>-tc<cols>-r<rank>-s<seed>`
and `gram-tr<rows>-tc<cols>` directories, preserving existing square variants.
The source dataset is never copied through CSV. Conversion and checksums
are recorded in `manifest.json`; only variants needed by active PreVision arms are
prepared. The timed run's `prevision-work` directory is deleted afterward, while
the compact JSON summary and a bounded engine log remain. The normal output
retention check keeps full numeric outputs only if executions fail or disagree.

The two large pilot cases have **not** yet been performance-validated on the remote
host. First compare results and cgroup I/O, then tune the PreVision tile/cache
settings if they show memory pressure or source-read amplification.

## Local GNMF tiling check (2026-09-30)

The same 32 GB FP64 matrix (4,000,000 × 1,000), rank 16, three updates, and
16 GiB cgroup were used throughout. PreVision had a 10 GiB data cache, four
BLAS threads, one elementwise thread, and the `memfd` arena. Preparation was
excluded; elapsed time includes factor export. New layouts each had two
cold-cache repetitions, with medians below.

| Input tile shape | Tile bytes | Input tiles | Elapsed time |
| --- | --- | --- | --- |
| 1,000 × 1,000 (previous single run) | 8 MB | 4,000 | 117.2 s |
| 20,000 × 1,000 | 160 MB | 200 | 42.1 s |
| 40,000 × 1,000 (new default) | 320 MB | 100 | 40.3 s |
| 80,000 × 1,000 | 640 MB | 50 | 40.5 s |

The 320 MB layout reduced CPU time from 378 to 132 seconds while physical reads
remained approximately 100 GB. Full W/H outputs agreed with the previous layout
to maximum absolute error 3.4e-15. No run failed or exceeded its memory limit.
The previous matched local controls took 31.4 seconds for SystemDS OOC and
41.6 seconds for Dask; these controls were not rerun for the tiling check.
Thus the larger tiles removed most of the earlier PreVision deficit, but do not
establish a significant advantage over Dask. This is a local pilot, not the
paper's different dataset shape, rank, update order, hardware, or memory budget.

Raw measurements, numerical checks, and bounded logs are retained under
`data_dir/agent_scratch/pvtiles_20260930_VHSw/` (`comparison.json`,
`tile-results.json`, and `results/`). Temporary native layouts and validated
factor outputs were removed; existing canonical data and older prepared variants
were preserved. The updated plan prepares its geometry-qualified variant on demand.
