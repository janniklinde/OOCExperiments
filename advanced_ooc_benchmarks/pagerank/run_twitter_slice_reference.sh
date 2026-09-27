#!/usr/bin/env bash
# One heap-capped PageRank reference run on the induced Twitter subgraph.
set -euo pipefail

mode="${1:?mode must be cp, ooc, or spark}"
blocksize="${2:?blocksize required}"
iterations="${3:?iteration count required}"
results="${4:?result directory required}"
[[ "$mode" == cp || "$mode" == ooc || "$mode" == spark ]] || {
  echo "mode must be cp, ooc, or spark" >&2; exit 2;
}
[[ "$blocksize" =~ ^[1-9][0-9]*$ && "$iterations" =~ ^[1-9][0-9]*$ ]] || {
  echo "blocksize and iterations must be positive integers" >&2; exit 2;
}

dataset="${TWITTER_SLICE_DATASET:-/workspace/data_dir/bench-data/twitter-2010-window-12m-2154m}"
systemds_jar="${SYSTEMDS_JAR:-/workspace/systemds/target/SystemDS.jar}"
java_bin="${JAVA_BIN:-/opt/devcon/env/java/current/bin/java}"
java_heap="${JAVA_HEAP:-12g}"
java_direct_memory="${JAVA_DIRECT_MEMORY:-2g}"
threads="${THREADS:-16}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
graph="$dataset/systemds/G-bs$blocksize"
dangling="$dataset/systemds/dangling-bs$blocksize"
[[ -s "$systemds_jar" && -s "$graph.mtd" && -s "$dangling.mtd" ]] || {
  echo "missing jar or native graph for blocksize $blocksize" >&2; exit 2;
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$results/java-tmp" "$results/systemds-tmp" "$results/systemds-scratch" "$results/output"
config="$results/SystemDS-config.xml"
cat >"$config" <<EOF
<root>
  <sysds.localtmpdir>$results/systemds-tmp</sysds.localtmpdir>
  <sysds.scratch>$results/systemds-scratch</sysds.scratch>
  <sysds.defaultblocksize>$blocksize</sysds.defaultblocksize>
  <sysds.ooc.io.direct>true</sysds.ooc.io.direct>
  <sysds.ooc.write.empty.blocks>true</sysds.ooc.write.empty.blocks>
  <sysds.ooc.io.reader.buffersize>1048576</sysds.ooc.io.reader.buffersize>
  <sysds.ooc.io.reader.threads>16</sysds.ooc.io.reader.threads>
  <sysds.ooc.io.reader.poolsize>16</sysds.ooc.io.reader.poolsize>
  <sysds.ooc.materialized.partition.bytes>16777216</sysds.ooc.materialized.partition.bytes>
  <sysds.ooc.cache.pack.bytes>0</sysds.ooc.cache.pack.bytes>
  <sysds.ooc.memory.broker.fraction>0.3333333333333333</sysds.ooc.memory.broker.fraction>
  <sysds.ooc.memory.broker.max>4294967296</sysds.ooc.memory.broker.max>
  <sysds.ooc.memory.prefetch.fraction>0.1</sysds.ooc.memory.prefetch.fraction>
  <sysds.ooc.memory.prefetch.max>1073741824</sysds.ooc.memory.prefetch.max>
  <sysds.ooc.memory.cache.fraction.soft>0.5</sysds.ooc.memory.cache.fraction.soft>
  <sysds.ooc.memory.cache.fraction.hard>0.6</sysds.ooc.memory.cache.fraction.hard>
  <sysds.ooc.source.replay.memory>1073741824</sysds.ooc.source.replay.memory>
  <sysds.ooc.source.bulksize>1073741824</sysds.ooc.source.bulksize>
  <sysds.ooc.source.replay.prefetch>4</sysds.ooc.source.replay.prefetch>
</root>
EOF

extra=()
if [[ "$mode" == ooc ]]; then extra=(-ooc -oocStats); fi
cd "$here/.."
TIMEFORMAT=$'wall_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
if [[ "$mode" == spark ]]; then
  export SPARK_HOME="${SPARK_HOME:-/opt/devcon/env/python/lib/python3.11/site-packages/pyspark}"
  export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$java_bin")")}" 
  export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
  spark_submit="${SPARK_SUBMIT:-/opt/devcon/env/python/bin/spark-submit}"
  spark_direct_memory="${SPARK_JAVA_DIRECT_MEMORY:-6g}"
  mkdir -p "$results/spark-local"
  java_opts="-XX:+UseG1GC -XX:G1HeapRegionSize=32m -XX:ActiveProcessorCount=$threads -XX:MaxDirectMemorySize=$spark_direct_memory --add-modules=jdk.incubator.vector -Djava.io.tmpdir=$results/java-tmp"
  for package in java.nio java.io java.util java.lang java.lang.ref java.lang.invoke java.util.concurrent sun.nio.ch; do
    java_opts+=" --add-opens=java.base/$package=ALL-UNNAMED"
  done
  { time timeout "$timeout_seconds" \
    "$spark_submit" --master "local[$threads]" \
    --driver-memory "$java_heap" --executor-memory "$java_heap" \
    --driver-java-options "$java_opts" \
    --conf "spark.local.dir=$results/spark-local" \
    --conf spark.ui.enabled=false \
    --conf spark.driver.maxResultSize=0 \
    --conf "spark.default.parallelism=$threads" \
    --conf "spark.sql.shuffle.partitions=$threads" \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.memory.fraction=0.25 \
    --conf spark.hadoop.fs.local.block.size=134217728 \
    "$systemds_jar" -f "$here/systemds.dml" -exec spark \
    -config "$config" -stats \
    -args "$graph" "$dangling" 0.85 "$iterations" "$results/output/rank" \
    >"$results/systemds.log" 2>&1; } 2>"$results/time.txt"
else
  { time timeout "$timeout_seconds" \
    "$java_bin" -Xms1g -Xmx"$java_heap" -XX:+UseG1GC -XX:G1HeapRegionSize=32m \
    -XX:ActiveProcessorCount="$threads" -XX:MaxDirectMemorySize="$java_direct_memory" \
    --add-modules=jdk.incubator.vector -Djava.io.tmpdir="$results/java-tmp" \
    -jar "$systemds_jar" -f "$here/systemds.dml" -exec singlenode \
    -config "$config" "${extra[@]}" -stats \
    -args "$graph" "$dangling" 0.85 "$iterations" "$results/output/rank" \
    >"$results/systemds.log" 2>&1; } 2>"$results/time.txt"
fi
if [[ ! -s "$results/output/rank.mtd" ]] || grep -q 'An Error Occurred' "$results/systemds.log"; then
  echo "SystemDS did not produce a valid rank output; inspect $results/systemds.log" >&2
  exit 1
fi
