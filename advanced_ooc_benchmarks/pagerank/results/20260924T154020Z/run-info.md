# PageRank iteration sweep

The input is the published-order induced Twitter-2010 interval
`[12,000,000, 21,540,000)` (9,540,000 vertices; 235,054,121 retained,
renormalized FP64 edges). CP and OOC read the same native
`G-bs10000` and `dangling-bs10000` files. Both execute
`pagerank/systemds.dml` with `alpha=0.85`, 16 visible CPU threads, a 14 GiB
Java heap, a 6 GiB direct-memory ceiling, and a binary rank output.

Each backend/iteration pair was run once, sequentially in the order
CP then OOC for 1, 3, 5, 10, 15, 30, 50, 75, and 100 iterations. The
30-100 iteration points were appended later on the same host. `Elapsed Time [s]` is
end-to-end Bash `time` wall time; `CPU Time [s]` is its user plus system
time. The raw logs and outputs are in
`/workspace/data_dir/pagerank-iteration-results/20260924T154020Z`.
All nine CP/OOC output pairs differed by at most `1.37e-15` in any rank
element. Every OOC run reported zero cache reloads and zero eviction writes.

The host lacked delegated benchmark cgroups and page cache was not dropped.
These are local reference measurements, not isolated cold-cache paper runs.
The SystemDS JAR SHA-256 was
`292fd289c619d0b3db9e0ff1c160e3e6ee147e6e7336b9ad01db704c79f9864d`.
