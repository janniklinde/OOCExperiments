# Operator-chain measurements

## Five-policy sweep

Medians of three measured iterations, following 30 seconds of JVM warmup. See
[the setup and limitations](operator-chains.md). GB is decimal. Cache capacity
is separate from the task allowance. Reorder costs are additional to preparation.

| Algorithm | Shape | Rank/k | Cache MiB | Policy | Iteration s | CPU s | Reload GB | Write GB | Tasks | Prep s | Reorder s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gnmf | 2000000 × 2 | 64 | 16 | arrival | 1.058 | 7.79 | 2.099 | 1.020 | 130 | 0.316 | 0.000 |
| gnmf | 2000000 × 2 | 64 | 16 | batch | 1.159 | 9.65 | 2.099 | 1.020 | 126 | 0.306 | 0.000 |
| gnmf | 2000000 × 2 | 64 | 16 | reorder | 1.035 | 7.68 | 2.098 | 1.019 | 126 | 0.323 | 0.471 |
| gnmf | 2000000 × 2 | 64 | 16 | single | 1.328 | 6.75 | 2.100 | 1.023 | 4000 | 0.323 | 0.000 |
| gnmf | 2000000 × 2 | 64 | 16 | window | 1.099 | 7.63 | 2.100 | 1.020 | 354 | 0.320 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | arrival | 0.857 | 6.25 | 2.291 | 1.022 | 504 | 0.325 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | batch | 1.523 | 20.16 | 2.296 | 1.023 | 500 | 0.322 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | reorder | 0.820 | 5.85 | 2.292 | 1.021 | 500 | 0.339 | 0.456 |
| gnmf | 8000000 × 2 | 16 | 16 | single | 1.978 | 11.82 | 2.294 | 1.024 | 16000 | 0.361 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | window | 0.993 | 6.41 | 2.323 | 1.024 | 1412 | 0.374 | 0.000 |
| kmeans | 8000000 × 2 | 32 | 16 | arrival | 0.451 | 2.09 | 0.116 | 0.000 | 252 | 0.051 | 0.000 |
| kmeans | 8000000 × 2 | 32 | 16 | batch | 0.608 | 5.00 | 0.118 | 0.000 | 250 | 0.062 | 0.000 |
| kmeans | 8000000 × 2 | 32 | 16 | reorder | 0.455 | 1.90 | 0.116 | 0.000 | 250 | 0.051 | 0.066 |
| kmeans | 8000000 × 2 | 32 | 16 | single | 0.886 | 3.59 | 0.118 | 0.000 | 8000 | 0.066 | 0.000 |
| kmeans | 8000000 × 2 | 32 | 16 | window | 0.465 | 2.03 | 0.116 | 0.000 | 706 | 0.056 | 0.000 |
| lmcg | 32000000 × 2 | 1 | 16 | arrival | 0.192 | 0.83 | 0.502 | 0.000 | 1002 | 0.171 | 0.000 |
| lmcg | 32000000 × 2 | 1 | 16 | batch | 0.825 | 13.01 | 0.508 | 0.000 | 1000 | 0.197 | 0.000 |
| lmcg | 32000000 × 2 | 1 | 16 | reorder | 0.184 | 0.89 | 0.502 | 0.000 | 1000 | 0.184 | 0.207 |
| lmcg | 32000000 × 2 | 1 | 16 | single | 1.472 | 6.12 | 0.506 | 0.000 | 32000 | 0.197 | 0.000 |
| lmcg | 32000000 × 2 | 1 | 16 | window | 0.303 | 1.59 | 0.502 | 0.000 | 2824 | 0.186 | 0.000 |

Checksums agree across policies for 12 iteration/shape combinations.
Total /proc storage reads across measured iterations: 81920 bytes.

## Streaming and resident controls

The resident controls include one initial reload and two resident passes; use
the subsequent ten-iteration follow-up to assess steady resident performance.
For GNMF lookahead, iteration 1 bootstraps statistics and the final iteration
omits unused next-iteration statistics. Its three-iteration median is therefore
not an end-to-end speedup estimate; compare total elapsed or interior iterations.

| Algorithm | Shape | Rank/k | Cache MiB | Policy | Iteration s | CPU s | Reload GB | Write GB | Tasks | Prep s | Reorder s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gnmf | 1000000 × 128 | 16 | 16 | arrival | 0.642 | 3.08 | 2.548 | 0.126 | 2000 | 0.362 | 0.000 |
| gnmf | 1000000 × 128 | 16 | 16 | lookahead | 0.464 | 2.13 | 1.256 | 0.126 | 1000 | 0.358 | 0.000 |
| gnmf | 2000000 × 2 | 64 | 16 | lookahead | 0.899 | 5.74 | 1.053 | 1.019 | 64 | 0.323 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | lookahead | 0.723 | 4.92 | 1.150 | 1.023 | 252 | 0.391 | 0.000 |
| kmeans | 8000000 × 2 | 32 | 16 | barrier | 1.290 | 4.55 | 0.382 | 0.128 | 506 | 0.053 | 0.000 |
| lmcg | 32000000 × 2 | 1 | 16 | barrier | 0.328 | 2.08 | 1.267 | 0.257 | 2004 | 0.168 | 0.000 |
| lmcg | 8000000 × 2 | 1 | 256 | arrival | 0.031 | 0.16 | 0.000 | 0.000 | 253 | 0.093 | 0.000 |
| lmcg | 8000000 × 2 | 1 | 256 | batch | 0.060 | 0.25 | 0.000 | 0.000 | 250 | 0.100 | 0.000 |
| lmcg | 8000000 × 2 | 1 | 256 | single | 0.034 | 0.16 | 0.000 | 0.000 | 8000 | 0.098 | 0.000 |

Checksums agree across policies for 18 iteration/shape combinations.
Total /proc storage reads across measured iterations: 0 bytes.

## Longer runs and shape-aware matching

The GNMF 8M×2 and resident LMCG rows use ten iterations. GNMF 1M×128 uses
three iterations. `partner` drives matching from W packs instead of X packs;
`lookahead-partner` combines that with next-iteration statistics. `reorder-x`
rewrites only X, not the initial W. These are separate invocations from the
three-iteration controls above.

| Algorithm | Shape | Rank/k | Cache MiB | Policy | Iteration s | CPU s | Reload GB | Write GB | Tasks | Prep s | Reorder s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gnmf | 1000000 × 128 | 16 | 16 | lookahead-partner | 0.388 | 2.27 | 1.145 | 0.125 | 251 | 0.351 | 0.000 |
| gnmf | 1000000 × 128 | 16 | 16 | partner | 0.513 | 3.42 | 2.284 | 0.126 | 502 | 0.355 | 0.000 |
| gnmf | 1000000 × 128 | 16 | 16 | reorder-x | 0.638 | 3.06 | 2.394 | 0.126 | 2000 | 0.340 | 0.478 |
| gnmf | 1000000 × 128 | 16 | 16 | window | 0.648 | 3.17 | 2.400 | 0.127 | 2000 | 0.376 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | arrival | 0.840 | 6.03 | 2.291 | 1.022 | 504 | 0.345 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | lookahead | 0.674 | 4.50 | 1.149 | 1.022 | 253 | 0.348 | 0.000 |
| gnmf | 8000000 × 2 | 16 | 16 | reorder-x | 0.825 | 5.90 | 2.290 | 1.021 | 500 | 0.321 | 0.054 |
| lmcg | 8000000 × 2 | 1 | 256 | arrival | 0.023 | 0.11 | 0.000 | 0.000 | 251 | 0.083 | 0.000 |
| lmcg | 8000000 × 2 | 1 | 256 | batch | 0.021 | 0.10 | 0.000 | 0.000 | 250 | 0.090 | 0.000 |
| lmcg | 8000000 × 2 | 1 | 256 | single | 0.022 | 0.09 | 0.000 | 0.000 | 8000 | 0.089 | 0.000 |

Checksums agree across policies for 23 iteration/shape combinations.
Total /proc storage reads across measured iterations: 0 bytes.

For resident LMCG, medians over iterations 4–10 are 20.94 ms (single),
20.55 ms (batch), and 21.73 ms (arrival). All have zero reload bytes.
The first spill-backed pass in those same JVMs is 487.06, 253.07, and 65.44 ms,
respectively. This control does not support a meaningful resident performance
win from packing for this kernel.

For GNMF 8M×2 rank 16, the ten-iteration totals are:

| Policy | Iteration total s | Additional reorganization s | Reload GB | Write GB |
|---|---:|---:|---:|---:|
| Arrival packs | 8.535 | 0 | 23.047 | 10.212 |
| Reorganize X only | 8.290 | 0.054 | 23.060 | 10.209 |
| Lookahead | 7.047 | 0 | 12.771 | 10.221 |

The first lookahead iteration bootstraps the statistics; the last avoids
computing unused future statistics. Interior-iteration medians are 0.839 s
(arrival) and 0.674 s (lookahead). The total saving is 17.4%, not the larger
saving one would infer from comparing the final iteration alone.

Directory inspection in the ten-iteration arrival run reports 252 X packs and
2,002 initial W packs: 2,459 true partition pairs, with 441 W packs touching
multiple X packs. After the first W update there are 2,001 W packs and 2,002
true pairs; only one W pack touches multiple X packs. Subsequent iterations
retain that layout. No inspection reads were needed: these counts come from
the primitive integer membership directories outside timed regions.

Raw logs are under `/workspace/data_dir/partition-chains-20260926`,
`partition-chains-controls-20260926`, and `partition-chains-followups-20260926`.
The consolidated [CSV](operator-chain-results.csv) preserves every phase and
identifies its source log and cohort. There are 39 completed benchmark
invocations and 159 measured iterations. Checksums agree across policies for
all corresponding shapes and iteration indices. Only 81,920 storage-read bytes
were recorded across all measured iterations; practically all reloads hit the
OS page cache. **Do not interpret reload GB / runtime as SSD throughput.**

