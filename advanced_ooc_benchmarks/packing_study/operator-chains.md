# Partition-aware operator-chain study

This is a test-only SystemDS prototype, not a planner integration or a result
from the full benchmark DML scripts. The implementation is
`systemds/src/test/java/org/apache/sysds/test/component/ooc/PartitionChainStudyTest.java`.

## What executes

- LMCG normal operator: `q = t(X) %*% (X %*% p)`. This repeats the expensive
  operator with fixed p; it is not a complete conjugate-gradient solver.
- Kmeans: distance matrix, row-minimum/tie-preserving normalized assignments,
  counts, and centroid sums. Three actual centroid updates, with k=32 and
  initial centers taken from the first k rows of X.
- GNMF: `H *= (W'X)/(W'W H + eps)`, then
  `W *= (X H')/(W H H' + eps)`. The H reduction is a real global barrier;
  each new W is packed into a new spillable store for the next iteration.

The custom chain processes row-local work while retaining the input lease.
LMCG does not store Xp; kmeans does not store D or P. GNMF must store the new W.
These are possible specialized/fused operators, not evidence that the present
generic primitive DAG automatically eliminates those intermediates.

Tasks reserve their complete inputs and output/scratch bound before requesting
cache entries. Four lanes overlap asynchronous cache requests and compute.
Compute workers never wait for disk or output handoff. Output packing uses
the existing concurrent PackingMaterializer, including asynchronous publication.
Only closed packs enter the PartitionedStore. Source preparation uses four
producers, each reversing 17-tile chunks; it is not perfect pre-partitioning.

## Policies

| Policy | Physical cache entries | Task unit | Locality |
|---|---|---|---|
| single | One-member packs | One tile | Arrival order |
| batch | One-member packs | Up to 32 entries | Arrival order |
| arrival | Up to 32 tiles / 512 KiB | One X pack and exact matching W packs | Arrival order |
| window | Same byte/count caps | Same | Close on changing 32-tile row window |
| reorder | Read and rewrite existing arrival-order stores | Same | Deliberately contiguous row packs |

The exact tile-to-pack directory is a primitive int array. Matching does not
load bounding-box false positives. This metadata is charged to the metadata
allowance. Matching partitions need not have identical boundaries. No cache
eviction policy was changed. Open builders are pre-reserved, not mutable
cache-managed packs.

## Measurement contract

Each JVM gets 30 seconds of repeated small-run warmup, a 3 GiB heap, four compute
threads, and the default 16 cache I/O readers. The spill experiments use a
16 MiB cache, separately from a 256 MiB task allowance and 64 MiB producer and
metadata allowances. Thus 16 MiB is **not** a process memory limit.
Maximum admitted atomic task size is 64 MiB. Logical blocksize is 1000.

All data and kernels are FP64. Dense synthetic X has two columns; GNMF ranks
16 and 64 intentionally stress factor expansion rather than represent a useful
low-rank approximation of a two-column matrix. This distinction matters when
extrapolating to plausible end-to-end workloads.

Preparation and deliberate reorganization are timed separately, including a
forced cache drain. Iterations are sequential and do not artificially flush
the cache between them. Cache read/write byte counters measure serialized
spill I/O. `/proc/self/io` separately records storage-accounted bytes. Buffered
I/O, no OS page-cache reset, and no cgroup isolation: these are **not** SSD
bandwidth measurements. Writeback can cross phase boundaries.

The forced-spill test checks all five policies on all three chains, two
iterations and a tail tile, against whole-matrix numerical calculations.
It checks actual spill writes, reloads, and release of task/producer ownership.

## Reproduction

Build the focused test in the SystemDS tree:

```bash
mvn -q -Dtest=PartitionChainStudyTest test
```

Then from OOCExperiments:

```bash
bash advanced_ooc_benchmarks/packing_study/run_partition_chains.sh \
  lmcg arrival 32000000 1 16777216 3 30
```

Arguments are algorithm, policy, rows, rank/k, cache bytes, iterations, warmup
seconds, and optional column count (default 2). Runtime-generated spill files are removed by fixture cleanup; existing
benchmark datasets are untouched. Results and limitations follow below.

## Progress failures found during warmup

Two important limitations surfaced before the larger measurements:

1. A one-shot forced cache drain after replacing an old store sometimes stopped
   with `owned=519512`, `producer=0`, `work=0`. The cache dump showed one entry,
   zero pins, zero deferred unpins, zero evicting bytes, and an eviction limit
   of zero. Requesting eviction again allowed completion. The test-only drain
   helper now retries `updateLimits(capacity, 0)` while waiting. Raw diagnostics:
   `/workspace/data_dir/chain-reorder-diagnostic2.log`. This demonstrates a
   progress problem; it does not identify the precise concurrency race.
2. Repeated GNMF arrival-order warmup with only a 1 MiB cache hit a 60-second
   timeout in `PackingMaterializer.Lane.append(...).get(...)` during preparation.
   Closing reported `Close requires all lane handoffs to finish`. This failure
   occurred before any large measured iteration, not in an algorithm kernel.
   Subsequent warmups use 16 MiB. Earlier successful controls used 1 MiB warmup;
   their measured cache was already 16 MiB. The 1 MiB forced-spill correctness
   tests still pass, but this is not sufficient evidence of long-run liveness.

No production cache changes were made to work around these issues. In
particular, the benchmark does not periodically prod eviction during timed
algorithm iterations. Reliable publication/drain progress remains an integration
requirement, separate from the policy performance conclusions.

## What these experiments establish

### 1. Packing and task batching are not substitutes

The LMCG normal-operator scan improves from 1.472 s to 0.825 s with scheduling
batches and to 0.192 s with physical packs. Read volume remains about 0.5 GB.
The resident ten-iteration control converges to about 21 ms for all three
policies on 128 MB. Thus the big measured advantage is on the spill-retrieval
path, not simply avoiding individual tasks for resident tiles.

Batching also increases the number of independently requested cache entries
in flight. It is not a pure task-queue microbenchmark: the batch-only CPU cost
can increase because 16 I/O threads service many more concurrent small requests.
These measurements isolate the combined entry/retrieval/deserialization path;
they do not assign exact percentages to cache locks versus serialization.

At rank 64, W tiles already contain about 512 KB of dense payload. The gain of
physical packing over task batching is correspondingly smaller (1.058 vs
1.159 s) than at rank 16 (0.857 vs 1.523 s).

### 2. Retaining row-local inputs matters more than strict layout

For kmeans, materializing P despite using packs costs 1.290 s, 0.382 GB reload,
and 0.128 GB writes per iteration. The input-retaining chain costs 0.451 s,
0.116 GB reload, and no intermediate writes. LMCG similarly changes from
0.328 s / 1.267 GB reload / 0.257 GB writes with stored Xp to
0.192 s / 0.502 GB reload / zero writes.

The assignment values remain FP64; their sparse serialization explains why the
kmeans assignment spill is much smaller than its dense allocated dimensions.
No quantization is involved.

A partition-aware task must be able to retain its input lease while applying
a row-local chain and contributing to small reductions. Transparent packing
followed by immediate per-tile unpacking and materialization would miss this
benefit. These prototypes implement that as a custom operator chain; normal
OOC DAG execution was not changed.

### 3. Exact membership and choice of driving input can replace reorganization

For GNMF 1M×128/rank16, X packs have one tile while W packs hold several.
Driving one task from each X pack causes repeated W-pack requests. Driving
from W instead gathers the matching X entries asynchronously into one admitted
unit and processes all W members. Neither side is physically reorganized.

| Policy | Iteration s | Reload GB | One-time reorganization s |
|---|---:|---:|---:|
| X-driven arrival packs | 0.642 | 2.548 | 0 |
| X-driven row windows | 0.648 | 2.400 | 0 |
| X-driven reorganized X | 0.638 | 2.394 | 0.478 |
| W-driven arrival packs | 0.513 | 2.284 | 0 |

The data imply a nearly single scan of both operands per phase for W-driven
matching. Its task count is about 500 per iteration instead of 2,000.
It does not assume equal partition IDs or matching partition boundaries.
Each task resolves actual partners through the membership directory.

This suggests an operator-local decision: inspect partition membership,
partner bytes, output expansion, and the atomic budget, then choose the driving
side and group matches. A whole-matrix repartitioning decision is not required
for this improvement.

### 4. Updated outputs should inherit affinity before considering a rewrite

The measured GNMF membership counts show that newly generated W naturally
becomes closely aligned with the X packs driving its computation. This removes
nearly all multi-X W packs after one iteration in the thin case.

Reorganizing both initial X and W cost 0.456 s for rank16, while W is replaced
immediately. Reorganizing only X cost 0.054 s in the ten-iteration follow-up;
its total advantage, including reorganization, was only about 2.2%. That is
too small, with this sample and runtime variability, to justify an unconditional
rewrite. The wider case likewise shows no compelling benefit from rewriting X.

The useful distinction is immutable reusable input versus repeatedly replaced
factor versus short-lived intermediate. Give output packers an inexpensive
affinity key from the producing task or row band. Do not infer that every
loop-carried matrix should receive expensive strict partitioning.

### 5. Pass scheduling is another lever, separate from packing

GNMF can compute the next sufficient statistics while Wnew and X are already
available:

```
A = W'X; B = W'W                         # bootstrap
repeat:
    H = H * A / (B H + eps)              # global small-matrix update
    stream matching X, W:
        Wnew = W * (X H') / (W H H' + eps)
        contribute Wnew'X and Wnew'Wnew  # next A and B, except final iteration
        publish Wnew to the packed store
    finish reductions; replace W
```

This preserves the global H dependency and still stores every new W. It does
not require W to fit RAM. On 8M×2/rank16, ten iterations fall from 8.535 s to
7.047 s; reload falls from 23.047 to 12.771 GB, while writes stay near 10.2 GB.
On 1M×128, combining this with W-driven matching takes 1.299 s for three
iterations versus 1.904 s for the original X-driven packed chain.

This is an algorithmic scheduling specialization, not a packing-only result.
The packing contract should permit it, but implementing a general loop rewrite
is not necessary to validate the primitive interface.

## Minimal integration suggested by the evidence

1. Preserve the closed-pack store and primitive integer directory. Keep precise
   membership/indexing optional, but available to matched-input primitives.
   Bounding-box candidates alone are not sufficient for scattered packs.
2. Provide one reusable admission/lease helper for a driving partition plus
   bounded matching partitions. A primitive may select either input as driver.
   Charge scratch and output expansion, not only input bytes.
3. Support row-local map/small-matrix multiply plus local partial reductions
   within that admitted unit. Let consumers specialize this path; do not
   require every primitive to implement every layout.
4. Let the output materializer accept an affinity key and byte/count bounds.
   Publish only closed immutable packs. Preserve input affinity cheaply for
   updated outputs rather than sorting each new version.
5. Carry lifetime/reuse and coordinate-affinity hints from HOP/loop analysis;
   decide actual packing, driving side, and budget size at the OOC planner or
   primitive where geometry and capabilities are known. There is no evidence
   here for a mandatory cost-based full reorganization pass.
6. Fix and stress-test publication/eviction progress before enabling this
   production-wide. The tiny-cache failures are not resolved by these policy
   results. The prototype's explicit drain retry is not a production fix.

A concrete admission issue was also caught: copying 32 tiles of 1000×128
required 73,879,488 bytes and exceeded the 64 MiB atomic bound. Reorganization
now bounds its copy chunk using the matrix width and scratch/input/output
estimate; the 1M×128 case then completed. A new forced-spill numerical case
covers that shape. General matched-input operators still need bounded
subdivision/fallback when their actual partner set exceeds the atomic cap;
this prototype rejects such unsupported tasks rather than blocking.

## Validation and limits

- Focused command: `mvn -q -Dtest=PartitionChainStudyTest test`.
  One JUnit method runs 22 numerical forced-spill configurations, including
  tails, two-input matching, factor replacement, lookahead, and wide reorganization.
- 39 successful benchmark invocations, 159 measured iterations, with equivalent
  checksums across matching policies. Full numbers are in
  [operator-chain-results.md](operator-chain-results.md).
- Source matrices are generated and packed by concurrent tile producers.
  Native SequenceFile source scanning, cold-SSD throughput, cgroups, and a
  complete 32 GiB DML workload are **not** evaluated here.
- All matrices fit within one logical column tile. Multi-column row-band
  assembly, sparse PageRank, rank512 admission, cancellation/error recovery,
  and general planner integration remain unvalidated.
- Initial sweep warmups used one small iteration and a 1 MiB cache; after the
  reported progress failure warmup cache increased to 16 MiB. Follow-ups use
  two small iterations per warmup cycle to exercise resident and lookahead
  paths as well. Every invocation still warmed for at least 30 seconds.
- The reported task counter counts admitted input groups, not every executor
  continuation or I/O task.
- Production SystemDS classes, normal benchmark plans, and existing datasets
  were left unchanged by this study. All new execution code is in the test
  harness. Temporary spill data is cleaned up; logs and compact results remain.
