# Real-world dataset preparation

`prepare.sh` is the common entry point for externally published datasets used by
the OOC experiments. Dataset-specific URLs and expected geometry live in its
manifest `case`; format conversion is streaming and is not benchmarked.

## Twitter 2010

The first dataset is the directed Twitter follower crawl from Kwak et al.,
distributed by the Laboratory for Web Algorithmics:

- 41,652,230 vertices
- 1,468,365,182 arcs
- 1,548,949 dangling vertices
- original arc `x -> y` means that `y` follows `x`

Prepare it with:

```bash
./experiments/real_world/prepare.sh twitter-2010
```

The default destination is
`$BENCH_DATA_DIR/real-world/twitter-2010`. Downloads are resumable and checked
against LAW's MD5 manifest. The converter streams the original and transposed
WebGraph files and writes normalized CSR, COO, a dangling bitmap, and SystemDS
binary blocks without creating a text edge list.

Defaults can be overridden:

```bash
PREP_WORKERS=8 PREP_JAVA_HEAP=2g REAL_WORLD_BLOCKSIZE=400000 \
  ./experiments/real_world/prepare.sh twitter-2010 /mnt/nvme/twitter-2010
```

Set `PREP_GRAPHFRAMES=1` to additionally export GraphFrames CSV. This is off by
default because formatting 1.47 billion rows is slow and Parquet should replace
it for serious DataFusion measurements.

Run the prepared graph with:

```bash
ITERATIONS=15 REPETITIONS=3 \
  ./experiments/pagerank/run_cgroup.sh /mnt/nvme/twitter-2010 /mnt/nvme/results/twitter-2010
```

LAW's compression ordering puts 137 million arcs in one 100k diagonal tile, which cannot run
under the OOC memory budget. Preparation therefore applies a deterministic bijective vertex
renumbering before writing any benchmark format. This preserves PageRank up to the same permutation
while distributing storage-order hotspots; the mapping is recorded in `metadata.json`.
The runner reads the graph block size from `systemds/G.mtd`, so no separate
`PAGERANK_BLOCKSIZE` setting is required. All comparable implementations use
uniform dangling-mass redistribution and fixed iterations.

The graph is described in Haewoon Kwak, Changhyun Lee, Hosung Park, and Sue
Moon, “What is Twitter, a Social Network or a News Media?”, WWW 2010. Follow
LAW's dataset citation and redistribution terms when publishing results.
