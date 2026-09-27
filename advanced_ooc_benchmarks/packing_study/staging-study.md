# Bounded 1D staging experiment

This extends `PartitionChainStudyTest`, not the production OOC planner. The
test-only `StagedPacking` adapter wraps the existing `PackingMaterializer` and
publishes ordinary immutable packs into the same `PartitionedStore`.

## Policy and ownership

- Four independent producer lanes divide the staging payload capacity equally.
  `stage50` means **50 MiB of payload in total**, not per lane. Primitive-array
  metadata is separately conservatively reserved (at most about 12 MiB for the
  four lanes). Both are charged to the producer allowance; there is no uncharged
  collection of retained tiles. The producer allowance is 128 MiB for all
  policies, with fixed cache and work allowances across comparisons.
- Staging keeps references to existing FP64 tiles; it does not copy matrix
  values. Arrival transfers ownership out of the producing task's budget into
  the pre-admitted staging budget. The actual open/closed pack cache model is
  unchanged: staging itself is not spillable.
- Upon byte or descriptor pressure, sort a primitive `long[]` of `(row, slot)`
  keys. These workloads have one logical column tile, so row index is the 1D
  spatial ordering. Partition the sorted sequence into byte/count-bounded
  candidate groups (32 tiles / 512 KiB, or one oversized tile).
- Select compact groups by row span, releasing approximately half the payload.
  Do not select the incomplete final candidate merely because its range is
  small. Always include the group containing the oldest retained tile so
  scattered leftovers cannot remain forever. Retain the other references for
  the next cycle. This is a bounded heuristic, not globally optimal clustering.
- A flush emits several packs per sort. Each candidate has a distinct locality
  key, preventing the delegate from merging unrelated candidates. Finalization
  flushes all tails. A tile larger than the staging limit bypasses staging and
  still obeys the delegate's pack bound.
- All publication continuations are asynchronous. Source producers can block
  on bounded handoff, but their flush continuations must run on a separate
  executor. Compute/output producers already suspend asynchronously.

`stagefifo50` uses the same buffering/admission but does not spatially sort.
`stage4` and `stage16` vary total payload capacity. `stage50x` stages only initial
X; initial W and subsequent W updates retain ordinary arrival packing. `window`
is the earlier close-on-window-change policy, not this buffered policy.
`stage50init` stages both initial inputs but preserves arrival packing for W
updates. `stage50partner` stages inputs and updates, but drives matched GNMF
tasks from W packs rather than X packs, using the same exact membership index.

## Experimental coverage

The existing LMCG normal-operator, KMeans row-local chain, and two-phase GNMF
chain are unchanged. Matching inputs use the exact tile-to-pack directory and
reserve actual partner packs. Four compute lanes and 16 cache I/O readers,
3 GiB JVM heap. No algorithm-specific lookahead is added in this experiment.

Arrival orders: contiguous/reversed 17-tile chunks, ordered chunks, shuffled
within 256-tile windows, and globally shuffled tile indices. X and initial W
use different deterministic permutations for shuffled inputs. Production of
W updates follows the actual asynchronous primitive execution, not a fabricated
independent order. Updated W is packed within iteration timing; source X and
initial W packing are timed in preparation.

`run_staging_suite.py` warms all represented kernels/policies for at least
60 seconds before measurements, then runs shuffled case order in one JVM.
Each case creates a fresh cache, allowances, stores, and input matrices. Five
iterations per case; source preparation is separately reported. The `repeat`
suite changes the input seed and case order. This is not one fresh JVM per
case; `run_staging_study.sh` provides that alternative with 30 s warmup each.
Additional suite names are `partner` (initial-only sorting and join direction),
`large` (256M-row vectors and 4-GB factors), and `large-repeat` (write-heavy
controls in reversed/repeated order). The latter also has GC logging. Its
timings exposed I/O-related nonstationarity, discussed in the findings; these
cases should not be treated as clean evidence of a sorting speedup or slowdown.

Both use real SystemDS serialized spill/reload with buffered I/O and no OS
page-cache reset. `/proc/self/io` storage-accounted bytes are recorded separately.
Systemd user scopes are unavailable in this container. These are **not** cold
SSD bandwidth or cgroup-constrained end-to-end DML benchmarks.

## Validation and known boundaries

The focused test currently checks 40 additional staged numerical configurations
plus the 22 earlier configurations: tails, three chains, all four arrival
patterns, small staging capacity, oversized-tile bypass, repeated factor
replacement, forced spill/reload, and allowance release. Whole-matrix calculations
validate intermediate reductions and GNMF factor values, not just final sums.

An initial pressure pilot exposed a source-executor self-deadlock: all four
source producers waited for a publication whose continuation was queued to
the same four-thread pool. The corrected preparation path dispatches flush
continuations to the fixture's separate executor. The small test passing alone
was not enough to validate this path; the larger pressure pilot was rerun.

General cancellation/failure recovery, cache-managed mutable open packs, sparse
PageRank, multi-column-tile bands, and integration into live dependent DAGs are
not validated. There is explicit completion flushing, but no timer-driven or
downstream-demand-driven early flush. Such a policy is required before using
large staging buffers in arbitrary live streaming dependencies.

## Reproduction

```bash
# In ../systemds:
mvn -q -Dtest=PartitionChainStudyTest test
# In OOCExperiments:
python3 advanced_ooc_benchmarks/packing_study/run_staging_suite.py \
  ../data_dir/staging-main --suite main
python3 advanced_ooc_benchmarks/packing_study/run_staging_suite.py \
  ../data_dir/staging-controls --suite controls
python3 advanced_ooc_benchmarks/packing_study/run_staging_suite.py \
  ../data_dir/staging-repeat --suite repeat
python3 advanced_ooc_benchmarks/packing_study/summarize_staging.py \
  ../data_dir/staging-main ../data_dir/staging-controls ../data_dir/staging-repeat \
  --output advanced_ooc_benchmarks/packing_study/staging-results
```

Spill files are removed by fixture shutdown. No benchmark datasets or normal
benchmark plans are changed.
