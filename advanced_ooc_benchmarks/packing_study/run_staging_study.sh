#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
result_dir=${1:?Pass a new results directory}
suite=${2:-main}
mkdir -p "$result_dir"
run_case() {
  local algo=$1 rows=$2 cols=$3 rank=$4 cache=$5 order=$6 policy=$7 seed=$8
  local name="${algo}-${rows}x${cols}-r${rank}-c${cache}-${order}-${policy}-s${seed}"
  echo "Starting $name"
  if timeout 240 bash "$script_dir/run_partition_chains.sh" "$algo" "$policy" "$rows" "$rank" "$cache" 5 30 "$cols" "$order" "$seed" > "$result_dir/$name.log" 2>&1; then
    echo "Finished $name"
  else
    echo "FAILED $name ($?)"
  fi
}
if [[ $suite == main ]]; then
  for spec in lmcg:32000000:2:1 kmeans:8000000:2:32 gnmf:8000000:2:16 gnmf:1000000:128:16; do
    IFS=: read -r algo rows cols rank <<< "$spec"
    for order in chunk random; do
      for policy in arrival window stagefifo50 stage50; do
        run_case "$algo" "$rows" "$cols" "$rank" 16777216 "$order" "$policy" 42
      done
    done
  done
elif [[ $suite == controls ]]; then
  for order in ordered window256 random; do
    for policy in arrival stage4 stage16 stage50 stage50x; do
      run_case gnmf 2000000 2 16 16777216 "$order" "$policy" 73
    done
  done
  for policy in arrival window stagefifo50 stage50; do
    run_case lmcg 8000000 1 1 268435456 random "$policy" 73
    run_case gnmf 1000000 2 16 4194304 random "$policy" 73
  done
else
  echo "Unknown suite: $suite" >&2
  exit 2
fi
