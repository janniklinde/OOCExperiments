#!/bin/bash
set -euo pipefail

# shellcheck disable=SC1091
source sysds_1.conf   # adjust path

FILE=./datagen.dml

for id in {1..3}; do
  # build cmd
  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"          # or OOC, doesn’t matter for datagen
    -f "$FILE"
    -exec singlenode
    -args 10000 10000 1.0 "$id"
  )

  printf 'SETUP CMD: %q ' "${cmd[@]}"; echo
  output=$("${cmd[@]}")
  printf '%s\n' "$output"
done

for id in {1..1}; do
  # build cmd
  cmd=(
    "${SYSDS_CMD_COMMON[@]}"
    "$SYSDS_JAR_CP"          # or OOC, doesn’t matter for datagen
    -f "$FILE"
    -exec singlenode
    -args 1000000 1000 1.0 "$id"
  )

  printf 'SETUP CMD: %q ' "${cmd[@]}"; echo
  output=$("${cmd[@]}")
  printf '%s\n' "$output"
done