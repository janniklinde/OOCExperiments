#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

experiments=(
  pca
  pca_warm
  lmcg
  lmcg_warm
  kmeans
  kmeans_warm
)

for experiment in "${experiments[@]}"; do
  echo "==> Running ${experiment}"
  (
    cd "${script_dir}/${experiment}"
    ./run.sh
  )
done
