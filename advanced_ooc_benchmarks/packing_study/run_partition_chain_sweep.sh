#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
result_dir=${1:?Pass a new results directory}
mkdir -p "$result_dir"
for case in lmcg:32000000:1 kmeans:8000000:32 gnmf:8000000:16 gnmf:2000000:64; do
  IFS=: read -r algorithm rows rank <<< "$case"
  for policy in single batch arrival window reorder; do
    name="${algorithm}-${rows}-r${rank}-${policy}"
    if [[ -f "$result_dir/$name.log" ]] && [[ $(rg -c '^iteration,' "$result_dir/$name.log" || true) == 3 ]] && ! rg -q 'Exception|AssertionError' "$result_dir/$name.log"; then
      continue
    fi
    echo "Starting $name"
    timeout 240 bash "$script_dir/run_partition_chains.sh" "$algorithm" "$policy" "$rows" "$rank" 16777216 3 30 > "$result_dir/$name.log" 2>&1
    echo "Finished $name"
  done
done
