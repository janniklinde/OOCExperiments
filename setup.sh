#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1091
source sysds_1.conf   # adjust path

MATRIX_FILE=./datagen.dml
BLOBS_FILE=./datagen_blobs.dml

target_exists() {
  local target="$1"
  [[ -e "$target" || -e "${target}.mtd" ]]
}

run_systemds() {
  local -a cmd=("$@")
  printf 'SETUP CMD: %q ' "${cmd[@]}"; echo
  local output
  output=$("${cmd[@]}")
  printf '%s\n' "$output"
}

for id in {1..3}; do
  target="data/mat_10000x10000s1.0_${id}"
  if target_exists "$target"; then
    echo "SKIP: target already exists: $target"
    continue
  fi

  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"
    -f "$MATRIX_FILE"
    -exec singlenode
    -args 10000 10000 1.0 "$id"
  )
  run_systemds "${cmd[@]}"
done

for id in {1..2}; do
  target="data/mat_1000000x1000s1.0_${id}"
  if target_exists "$target"; then
    echo "SKIP: target already exists: $target"
    continue
  fi

  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"
    -f "$MATRIX_FILE"
    -exec singlenode
    -args 1000000 1000 1.0 "$id"
  )
  run_systemds "${cmd[@]}"
done

for id in {1..2}; do
  target="data/mat_1000000x1s1.0_${id}"
  if target_exists "$target"; then
    echo "SKIP: target already exists: $target"
    continue
  fi

  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"
    -f "$MATRIX_FILE"
    -exec singlenode
    -args 1000000 1 1.0 "$id"
  )
  run_systemds "${cmd[@]}"
done

mkdir -p data/blobs
blob_base="data/blobs/8gb"
if target_exists "${blob_base}_X" || target_exists "${blob_base}_C" || target_exists "${blob_base}_Y"; then
  echo "SKIP: blob target already exists: ${blob_base}_{X,C,Y}"
else
  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"
    -f "$BLOBS_FILE"
    -exec singlenode
    -args 4000000 256 32 1.0 0.1 7 "$blob_base"
  )
  run_systemds "${cmd[@]}"
fi
