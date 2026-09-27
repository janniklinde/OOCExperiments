#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
result_dir=${1:?Pass a results directory}
mkdir -p "$result_dir"
for case in \
  lmcg:barrier:32000000:1:16777216:2 \
  kmeans:barrier:8000000:32:16777216:2 \
  gnmf:lookahead:8000000:16:16777216:2 \
  gnmf:lookahead:2000000:64:16777216:2 \
  lmcg:single:8000000:1:268435456:2 \
  lmcg:batch:8000000:1:268435456:2 \
  lmcg:arrival:8000000:1:268435456:2 \
  gnmf:arrival:1000000:16:16777216:128 \
  gnmf:lookahead:1000000:16:16777216:128; do
  IFS=: read -r algorithm policy rows rank cache cols <<< "$case"
  name="${algorithm}-${rows}x${cols}-r${rank}-${policy}-cache${cache}"
  echo "Starting $name"
  timeout 240 bash "$script_dir/run_partition_chains.sh" "$algorithm" "$policy" "$rows" "$rank" "$cache" 3 30 "$cols" > "$result_dir/$name.log" 2>&1
  echo "Finished $name"
done
