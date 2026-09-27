#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
result_dir=${1:?Pass a results directory}
mkdir -p "$result_dir"
for case in \
  gnmf:arrival:8000000:16:16777216:2:10 \
  gnmf:lookahead:8000000:16:16777216:2:10 \
  gnmf:reorder-x:8000000:16:16777216:2:10 \
  lmcg:single:8000000:1:268435456:2:10 \
  lmcg:batch:8000000:1:268435456:2:10 \
  lmcg:arrival:8000000:1:268435456:2:10 \
  gnmf:window:1000000:16:16777216:128:3 \
  gnmf:reorder-x:1000000:16:16777216:128:3 \
  gnmf:partner:1000000:16:16777216:128:3 \
  gnmf:lookahead-partner:1000000:16:16777216:128:3; do
  IFS=: read -r algorithm policy rows rank cache cols iterations <<< "$case"
  name="${algorithm}-${rows}x${cols}-r${rank}-${policy}-cache${cache}"
  echo "Starting $name"
  timeout 240 bash "$script_dir/run_partition_chains.sh" "$algorithm" "$policy" "$rows" "$rank" "$cache" "$iterations" 30 "$cols" > "$result_dir/$name.log" 2>&1
  echo "Finished $name"
done
