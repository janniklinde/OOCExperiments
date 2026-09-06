#!/usr/bin/env bash
# Prepare a named real-world dataset for the OOC experiment suite.
# Usage: ./prepare.sh DATASET [OUTPUT_DIR]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Only PYTHON, SYSDS_JAR and BENCH_DATA_DIR are needed from the environment, and the
# benchmark plan passes all three in the dataset's `env:`. bench_config.sh is the legacy
# standalone path: source it only when it is present and the variables are not already set,
# so this script works both from the plan directory and from the old experiments/ tree.
# shellcheck source=../bench_config.sh
if [[ -z "${PYTHON:-}" || -z "${SYSDS_JAR:-}" ]] && [[ -f "$here/../bench_config.sh" ]]; then
  source "$here/../bench_config.sh"
fi
: "${PYTHON:=python3}"
[[ -n "${SYSDS_JAR:-}" ]] || {
  echo "SYSDS_JAR is not set and no bench_config.sh is available to derive it" >&2
  exit 2
}
dataset="${1:-}"
[[ -n "$dataset" ]] || { echo "usage: $0 DATASET [OUTPUT_DIR]" >&2; exit 2; }

case "$dataset" in
  twitter-2010)
    base_url="http://data.law.di.unimi.it/webdata/twitter-2010"
    vertices=41652230
    edges=1468365182
    blocksize="${REAL_WORLD_BLOCKSIZE:-400000}"
    permutation_multiplier=104729
    permutation_offset=12345
    ;;
  *)
    echo "Unknown real-world dataset: $dataset" >&2
    echo "Available datasets: twitter-2010" >&2
    exit 2
    ;;
esac

out="${2:-$BENCH_DATA_DIR/real-world/$dataset}"
raw="$out/webgraph"
mkdir -p "$raw" "$out/systemds"
: "${PREP_WORKERS:=$(nproc)}"
: "${PREP_CONCURRENCY:=$PREP_WORKERS}"
: "${PREP_JAVA_HEAP:=2g}"
: "${PREP_GRAPHFRAMES:=0}"
: "${PREP_SKEW_JAVA_HEAP:=20g}"
[[ "$PREP_WORKERS" =~ ^[1-9][0-9]*$ && "$PREP_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "PREP_WORKERS and PREP_CONCURRENCY must be positive integers" >&2
  exit 2
}

fetch() {
  local file="$1"
  [[ -s "$raw/$file" ]] && return
  echo "downloading $file" >&2
  curl -fL --retry 5 --continue-at - -o "$raw/$file.part" "$base_url/$file"
  mv "$raw/$file.part" "$raw/$file"
}

for file in "$dataset.graph" "$dataset.properties" "$dataset-t.graph" \
  "$dataset-t.properties" "$dataset.md5sums" "$dataset.stats"; do
  fetch "$file"
done
(
  cd "$raw"
  grep -E "  ${dataset}(-t)?\.(graph|properties)$" "$dataset.md5sums" | md5sum -c -
)

java_project="$here/java"
# The benchmark container ships this tree with target/ already built and carries no
# maven, so build only when the artifacts are missing and a maven is actually there.
if [[ -d "$java_project/target/classes" && -d "$java_project/target/dependency" ]]; then
  echo "using the prebuilt WebGraph converter in $java_project/target" >&2
elif command -v mvn >/dev/null; then
  echo "building streaming WebGraph converter" >&2
  mvn -ntp -q -f "$java_project/pom.xml" package dependency:copy-dependencies \
    -DoutputDirectory=target/dependency
else
  echo "no prebuilt converter in $java_project/target and no maven to build one" >&2
  exit 2
fi
webgraph_cp="$java_project/target/classes:$java_project/target/dependency/*"
if [[ -s "$out/metadata.json" && -n "${permutation_multiplier:-}" ]]; then
  prepared_multiplier="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("vertex_permutation", {}).get("multiplier", ""))' "$out/metadata.json")"
  if [[ "$prepared_multiplier" != "$permutation_multiplier" ]]; then
    echo "existing derived data uses a different vertex ordering; regenerating it" >&2
    rm -rf "$out/csr" "$out/coo" "$out/systemds" "$out/dangling.u8" "$out/metadata.json"
    mkdir -p "$out/systemds"
  fi
fi
if [[ ! -s "$out/metadata.json" ]]; then
  if [[ -n "${permutation_multiplier:-}" && ! -s "$raw/$dataset-t.offsets" ]]; then
    echo "generating transpose offsets for deterministic vertex renumbering" >&2
    java -Xmx"$PREP_JAVA_HEAP" -cp "$webgraph_cp" it.unimi.dsi.webgraph.BVGraph --offsets "$raw/$dataset-t"
  fi
  converter_args=("$raw/$dataset" "$raw/$dataset-t" "$out" "$dataset")
  [[ -n "${permutation_multiplier:-}" ]] \
    && converter_args+=("$permutation_multiplier" "$permutation_offset")
  java -Xmx"$PREP_JAVA_HEAP" -cp "$webgraph_cp" WebGraphToPageRank "${converter_args[@]}"
fi

actual_vertices="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["vertices"])' "$out/metadata.json")"
actual_edges="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["edges"])' "$out/metadata.json")"
[[ "$actual_vertices" == "$vertices" && "$actual_edges" == "$edges" ]] || {
  echo "Prepared geometry $actual_vertices x $actual_edges does not match manifest $vertices x $edges" >&2
  exit 1
}

sysds_classes="$java_project/target/systemds-classes"
mkdir -p "$sysds_classes"
sysds_cp="$SYSDS_JAR:$(dirname "$SYSDS_JAR")/lib/*"
javac -cp "$sysds_cp" -d "$sysds_classes" "$java_project/systemds/CSRToSystemDS.java" \
  "$java_project/systemds/DanglingToSystemDS.java"

blocks=$(((vertices + blocksize - 1) / blocksize))
(( PREP_WORKERS > blocks )) && PREP_WORKERS=$blocks
native_suffix="${REAL_WORLD_NATIVE_SUFFIX:-}"
native_log_tag="${native_suffix:+${native_suffix}-}"
native_g="$out/systemds/G$native_suffix"
if [[ ! -s "$native_g.mtd" ]]; then
  rm -rf "$native_g"
  mkdir -p "$native_g"
  echo "writing $blocks x $blocks SystemDS blocks with $PREP_WORKERS workers" >&2
  # Twitter's compression ordering has one exceptionally dense diagonal region. At the default
  # 16-way split it belongs to worker 8 and needs ~15 GiB while serializing its largest tile. Run
  # that worker alone after the ordinary workers so aggregate preparation memory stays bounded.
  skew_worker=-1
  if [[ "$dataset" == twitter-2010 && "$blocksize" == 100000 ]]; then
    for ((worker=0; worker<PREP_WORKERS; worker++)); do
      first=$((worker * blocks / PREP_WORKERS))
      last=$(((worker + 1) * blocks / PREP_WORKERS))
      (( first <= 215 && 215 < last )) && skew_worker=$worker
    done
  fi
  pids=()
  failed=0
  for ((worker=0; worker<PREP_WORKERS; worker++)); do
    (( worker == skew_worker )) && continue
    first=$((worker * blocks / PREP_WORKERS))
    last=$(((worker + 1) * blocks / PREP_WORKERS))
    java -Xmx"$PREP_JAVA_HEAP" -cp "$sysds_classes:$sysds_cp" CSRToSystemDS "$out/csr" \
      "$native_g/part-$(printf '%05d' "$worker")" "$vertices" "$blocksize" "$first" "$last" \
      >"$out/systemds/convert-$native_log_tag$worker.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= PREP_CONCURRENCY )); then
      wait "${pids[0]}" || failed=1
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "SystemDS conversion failed; inspect $out/systemds/convert-*.log" >&2; exit 1; }
  if (( skew_worker >= 0 )); then
    first=$((skew_worker * blocks / PREP_WORKERS))
    last=$(((skew_worker + 1) * blocks / PREP_WORKERS))
    java -Xmx"$PREP_SKEW_JAVA_HEAP" -cp "$sysds_classes:$sysds_cp" CSRToSystemDS "$out/csr" \
      "$native_g/part-$(printf '%05d' "$skew_worker")" "$vertices" "$blocksize" "$first" "$last" \
      >"$out/systemds/convert-$native_log_tag$skew_worker.log" 2>&1
  fi
  cat > "$native_g.mtd" <<EOF
{
  "data_type": "matrix",
  "value_type": "double",
  "rows": $vertices,
  "cols": $vertices,
  "rows_in_block": $blocksize,
  "cols_in_block": $blocksize,
  "nnz": $edges,
  "format": "binary"
}
EOF
fi

native_dangling="$out/systemds/dangling$native_suffix"
if [[ ! -s "$native_dangling.mtd" ]]; then
  rm -rf "$native_dangling"
  mkdir -p "$native_dangling"
  java -Xmx"$PREP_JAVA_HEAP" -cp "$sysds_classes:$sysds_cp" DanglingToSystemDS "$out/dangling.u8" \
    "$native_dangling/part-00000" "$vertices" "$blocksize"
  dangling_nnz="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["dangling_vertices"])' "$out/metadata.json")"
  cat > "$native_dangling.mtd" <<EOF
{
  "data_type": "matrix",
  "value_type": "double",
  "rows": $vertices,
  "cols": 1,
  "rows_in_block": $blocksize,
  "cols_in_block": $blocksize,
  "nnz": $dangling_nnz,
  "format": "binary"
}
EOF
fi

if [[ "$PREP_GRAPHFRAMES" == 1 && ! -d "$out/graphframes-csv" ]]; then
  "$PYTHON" "$here/../pagerank/prepare_graphframes.py" "$out"
fi

cat <<EOF
Prepared $dataset in $out
  vertices      $vertices
  edges         $edges
  block size    $blocksize
  CSR/COO       $out/{csr,coo}
  SystemDS      $native_g
  dangling      $native_dangling
EOF
