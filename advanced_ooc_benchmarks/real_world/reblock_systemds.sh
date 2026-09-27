#!/usr/bin/env bash
# Reblock an existing binary matrix through SystemDS OOC into a new native dataset.
# Usage: reblock_systemds.sh SOURCE TARGET BLOCKSIZE
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_path="${1:?source binary matrix required}"
target_path="${2:?target binary matrix required}"
blocksize="${3:?target blocksize required}"
: "${SYSDS_JAR:?SYSDS_JAR is required}"
: "${JAVA:=java}"
: "${PREP_JAVA_HEAP:=20g}"
: "${PREP_THREADS:=16}"
: "${PREP_MIN_PARTS:=1}"
: "${PREP_EMIT_EMPTY_BLOCKS:=1}"
: "${PREP_CACHE_PACK_BYTES:=0}"
[[ "$blocksize" =~ ^[1-9][0-9]*$ && "$PREP_THREADS" =~ ^[1-9][0-9]*$ && "$PREP_MIN_PARTS" =~ ^[1-9][0-9]*$ ]] || {
  echo "blocksize, PREP_THREADS, and PREP_MIN_PARTS must be positive integers" >&2
  exit 2
}
[[ "$PREP_EMIT_EMPTY_BLOCKS" == 0 || "$PREP_EMIT_EMPTY_BLOCKS" == 1 ]] || {
  echo "PREP_EMIT_EMPTY_BLOCKS must be 0 or 1" >&2
  exit 2
}
[[ "$PREP_CACHE_PACK_BYTES" =~ ^[0-9]+$ ]] || {
  echo "PREP_CACHE_PACK_BYTES must be a nonnegative integer" >&2
  exit 2
}
[[ -s "$source_path.mtd" && -e "$source_path" ]] || {
  echo "missing source binary matrix: $source_path" >&2
  exit 2
}
[[ "$source_path" != "$target_path" ]] || {
  echo "source and target must differ" >&2
  exit 2
}

mkdir -p "$(dirname "$target_path")"
work_dir="$(mktemp -d "$(dirname "$target_path")/.reblock-XXXXXX")"
complete="$target_path.ooc-reblocked"
cleanup() { rm -rf "$work_dir"; }
trap cleanup EXIT
mkdir -p "$work_dir/tmp" "$work_dir/scratch"
if [[ "$PREP_EMIT_EMPTY_BLOCKS" == 1 ]]; then emit_empty=true; else emit_empty=false; fi
config="$work_dir/SystemDS-config.xml"
cat > "$config" <<EOF
<root>
  <sysds.localtmpdir>$work_dir/tmp</sysds.localtmpdir>
  <sysds.scratch>$work_dir/scratch</sysds.scratch>
  <sysds.defaultblocksize>$blocksize</sysds.defaultblocksize>
  <sysds.ooc.write.empty.blocks>$emit_empty</sysds.ooc.write.empty.blocks>
  <sysds.ooc.io.direct>true</sysds.ooc.io.direct>
  <sysds.ooc.io.reader.threads>$PREP_THREADS</sysds.ooc.io.reader.threads>
  <sysds.ooc.io.reader.poolsize>$PREP_THREADS</sysds.ooc.io.reader.poolsize>
  <sysds.ooc.cache.pack.bytes>$PREP_CACHE_PACK_BYTES</sysds.ooc.cache.pack.bytes>
</root>
EOF
output="$work_dir/output"
echo "OOC reblock: $source_path -> $target_path (blocksize=$blocksize, threads=$PREP_THREADS, empty=$emit_empty)" >&2
"$JAVA" -Xmx"$PREP_JAVA_HEAP" -XX:ActiveProcessorCount="$PREP_THREADS" \
  --add-modules=jdk.incubator.vector \
  -jar "$SYSDS_JAR" -f "$here/reblock_systemds.dml" -exec singlenode -ooc \
  -config "$config" -args "$source_path" "$output" "$blocksize"
[[ -s "$output.mtd" && -e "$output" ]] || {
  echo "SystemDS did not create a complete binary output" >&2
  exit 1
}
if [[ -d "$output" ]]; then
  part_count="$(find "$output" -maxdepth 1 -type f ! -name '.*' | wc -l)"
else
  part_count=1
fi
[[ "$part_count" -ge "$PREP_MIN_PARTS" ]] || {
  echo "SystemDS output has only $part_count part files (required: $PREP_MIN_PARTS)" >&2
  exit 1
}
echo "OOC reblock wrote $part_count part files" >&2
if [[ -e "$target_path" || -e "$target_path.mtd" || -e "$complete" ]]; then
  echo "target exists; refusing to overwrite: $target_path" >&2
  exit 1
fi
mv "$output" "$target_path"
mv "$output.mtd" "$target_path.mtd"
printf 'source=%s\nblocksize=%s\nparts=%s\nemit_empty=%s\n' \
  "$source_path" "$blocksize" "$part_count" "$emit_empty" > "$complete"
