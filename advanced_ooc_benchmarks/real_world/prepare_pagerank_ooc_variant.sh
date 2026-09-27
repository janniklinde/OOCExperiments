#!/usr/bin/env bash
# Convert the CSR graph to the requested tile size, then write benchmark files through SystemDS OOC.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="${1:?dataset directory required}"
blocksize="${2:?blocksize required}"
[[ "$blocksize" =~ ^[1-9][0-9]*$ ]] || { echo "invalid blocksize: $blocksize" >&2; exit 2; }
source_g="$dataset_dir/systemds/G-bs$blocksize"
source_dangling="$dataset_dir/systemds/dangling-bs$blocksize"
target_g="$dataset_dir/systemds/G-ooc-bs$blocksize"
target_dangling="$dataset_dir/systemds/dangling-ooc-bs$blocksize"

ready() {
  [[ -d "$1" && -s "$1.mtd" && -s "$1.ooc-reblocked" ]] &&
    grep -qx "blocksize=$blocksize" "$1.ooc-reblocked" &&
    grep -qx 'emit_empty=true' "$1.ooc-reblocked" &&
    awk -F= -v minimum="${PREP_THREADS:-16}" \
      '$1 == "parts" && $2 + 0 >= minimum {found = 1} END {exit !found}' "$1.ooc-reblocked"
}

if ready "$target_g" && ready "$target_dangling"; then
  exit 0
fi

# The 10k staging matrix is a base-plan artifact. Other staging sizes are generated
# on demand by the existing streaming CSR converter, but are never benchmarked directly.
if [[ "$blocksize" != 10000 ]] &&
  { { ! ready "$target_g" &&
      { [[ ! -s "$source_g.complete-blocks" ]] ||
        ! grep -qx "blocksize=$blocksize" "$source_g.complete-blocks"; }; } ||
    { ! ready "$target_dangling" &&
      [[ ! -s "$source_dangling.complete-blocks" ]]; }; }; then
  REAL_WORLD_BLOCKSIZE="$blocksize" REAL_WORLD_NATIVE_SUFFIX="-bs$blocksize" \
    "$here/prepare.sh" twitter-2010 "$dataset_dir"
fi

pack_bytes=0
if (( blocksize <= 2500 )); then
  # The 2.5k graph has roughly 278 million logical tiles; transient source packing
  # keeps preparation below the cache-entry metadata limit. Measured runs do not use it.
  pack_bytes=16777216
fi
if ! ready "$target_g"; then
  PREP_MIN_PARTS="${PREP_THREADS:-16}" PREP_CACHE_PACK_BYTES="$pack_bytes" "$here/reblock_systemds.sh" \
    "$source_g" "$target_g" "$blocksize"
fi
if ! ready "$target_dangling"; then
  PREP_MIN_PARTS=1 "$here/reblock_systemds.sh" "$source_dangling" \
    "$target_dangling" "$blocksize"
fi
