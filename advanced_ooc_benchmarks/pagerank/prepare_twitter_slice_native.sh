#!/usr/bin/env bash
# Convert a prepared induced-prefix CSR graph to complete SystemDS block grids.
set -euo pipefail

dataset_dir="${1:?dataset directory required}"
blocksize="${2:?blocksize required}"
systemds_jar="${SYSTEMDS_JAR:-/workspace/systemds/target/SystemDS.jar}"
java_home="${JAVA_HOME:-/opt/devcon/env/java/current}"
workers="${PREP_WORKERS:-16}"
concurrency="${PREP_CONCURRENCY:-2}"
java_heap="${PREP_JAVA_HEAP:-3g}"
[[ "$blocksize" =~ ^[1-9][0-9]*$ && "$workers" =~ ^[1-9][0-9]*$ &&
   "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "invalid numeric option" >&2; exit 2; }
[[ -s "$systemds_jar" && -s "$dataset_dir/metadata.json" ]] || {
  echo "missing SystemDS jar or prepared CSR dataset" >&2; exit 2;
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_java="$here/../real_world/java/systemds"
classes="$dataset_dir/java-classes"
mkdir -p "$classes" "$dataset_dir/systemds"
classpath="$systemds_jar:$(dirname "$systemds_jar")/lib/*"
"$java_home/bin/javac" -proc:none -cp "$classpath" -d "$classes" \
  "$source_java/CSRToSystemDS.java" "$source_java/DanglingToSystemDS.java"

vertices="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["vertices"])' "$dataset_dir/metadata.json")"
edges="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["edges"])' "$dataset_dir/metadata.json")"
dangling="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dangling_vertices"])' "$dataset_dir/metadata.json")"
blocks=$(((vertices + blocksize - 1) / blocksize))
(( workers > blocks )) && workers="$blocks"
graph="$dataset_dir/systemds/G-bs$blocksize"
dangling_matrix="$dataset_dir/systemds/dangling-bs$blocksize"
if [[ -s "$graph.mtd" && -s "$dangling_matrix.mtd" ]]; then
  echo "reusing $graph and $dangling_matrix"
  exit 0
fi
mkdir -p "$graph" "$dangling_matrix"

pids=()
failed=0
for ((worker=0; worker<workers; worker++)); do
  first=$((worker * blocks / workers))
  last=$(((worker + 1) * blocks / workers))
  part="$(printf '%05d' "$worker")"
  "$java_home/bin/java" -Xmx"$java_heap" -cp "$classes:$classpath" CSRToSystemDS \
    "$dataset_dir/csr" "$graph/part-$part" "$vertices" "$blocksize" "$first" "$last" true \
    >"$dataset_dir/systemds/convert-bs$blocksize-$part.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= concurrency )); then
    wait "${pids[0]}" || failed=1
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "graph conversion failed" >&2; exit 1; }

"$java_home/bin/java" -Xmx"$java_heap" -cp "$classes:$classpath" DanglingToSystemDS \
  "$dataset_dir/dangling.u8" "$dangling_matrix/part-00000" "$vertices" "$blocksize" true \
  >"$dataset_dir/systemds/convert-dangling-bs$blocksize.log" 2>&1

printf '{"data_type":"matrix","value_type":"double","rows":%s,"cols":%s,"rows_in_block":%s,"cols_in_block":%s,"nnz":%s,"format":"binary"}\n' \
  "$vertices" "$vertices" "$blocksize" "$blocksize" "$edges" >"$graph.mtd"
printf '{"data_type":"matrix","value_type":"double","rows":%s,"cols":1,"rows_in_block":%s,"cols_in_block":%s,"nnz":%s,"format":"binary"}\n' \
  "$vertices" "$blocksize" "$blocksize" "$dangling" >"$dangling_matrix.mtd"
echo "prepared $blocksize-block native graph in $graph"
