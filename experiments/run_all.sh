#!/bin/bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

experiments=(
  pca
  pca_warm
  lmcg
  lmcg_warm
  kmeans
  kmeans_warm
)

failed_experiments=()

for experiment in "${experiments[@]}"; do
  echo "==> Running ${experiment}"
  if (
    cd "${script_dir}/${experiment}"
    ./run.sh
  ); then
    echo "==> Finished ${experiment}"
  else
    echo "==> Failed ${experiment}, continuing"
    failed_experiments+=("${experiment}")
  fi
done

if ((${#failed_experiments[@]} > 0)); then
  echo "Completed with failures in: ${failed_experiments[*]}"
  exit 1
fi

echo "Completed all experiments successfully."
