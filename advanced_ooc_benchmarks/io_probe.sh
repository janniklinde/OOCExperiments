#!/usr/bin/env bash
# Measure what a candidate benchmark data root can actually deliver.
#
# Two questions, because they have different answers on an array: what does one
# reader get (which is what a single-threaded scan sees), and what does the device
# deliver in aggregate (which is what a 16-drive array is for). A single dd stream
# at queue depth 1 badly understates a wide array, so this sweeps stream counts.
#
# Everything uses O_DIRECT, so no root and no drop_caches is needed: the page cache
# is bypassed rather than flushed, which is also the only honest way to measure a
# device from inside a container.
set -euo pipefail

size_mb="${IO_PROBE_SIZE_MB:-4096}"
streams="${IO_PROBE_STREAMS:-1 4 16}"
keep=0

usage() {
  cat <<'USAGE'
Usage: ./io_probe.sh [options] PATH [PATH...]

  Probes each PATH (a directory; it will be created if missing).

  --size-mb N     bytes per stream, in MiB (default 4096; must exceed RAM cache
                  effects, and the total is N x max(streams))
  --streams "..." space-separated concurrency levels (default "1 4 16")
  --keep          leave the test files behind
  -h, --help      this message

Needs roughly (size-mb x max streams) MiB free per path, removed afterwards
unless --keep.
USAGE
}

paths=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --size-mb) size_mb="$2"; shift 2 ;;
    --streams) streams="$2"; shift 2 ;;
    --keep) keep=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done
[[ ${#paths[@]} -gt 0 ]] || { usage >&2; exit 2; }

max_streams=0
for n in $streams; do (( n > max_streams )) && max_streams=$n; done

topology() {
  local path="$1"
  echo "  free        $(df -h "$path" | awk 'NR==2 {print $4" of "$2" on "$1}')"
  if command -v findmnt >/dev/null; then
    echo "  mount       $(findmnt -n -o TARGET,SOURCE,FSTYPE -T "$path" | tr -s ' ')"
    local source
    source="$(findmnt -n -o SOURCE -T "$path" | sed 's/\[.*//')"
    if command -v lsblk >/dev/null && [[ -b "$source" || "$source" == /dev/* ]]; then
      # -s walks toward the physical devices, so an md or dm layer shows its members
      # and the drive count is visible rather than assumed.
      lsblk -s -o NAME,SIZE,ROTA,TYPE,MODEL "$source" 2>/dev/null | sed 's/^/  /' || true
    fi
  fi
  [[ -r /proc/mdstat ]] && grep -q '^md' /proc/mdstat 2>/dev/null &&
    { echo "  md arrays"; sed 's/^/    /' /proc/mdstat; }
  return 0
}

# One stream: dd with O_DIRECT, reporting MB/s from dd's own timing.
one_stream() {
  local file="$1" mode="$2"
  if [[ "$mode" == write ]]; then
    dd if=/dev/zero of="$file" bs=1M count="$size_mb" oflag=direct 2>&1
  else
    dd if="$file" of=/dev/null bs=1M count="$size_mb" iflag=direct 2>&1
  fi | awk '/copied|bytes/ {print $(NF-1)}'
}

probe() {
  local dir="$1" mode="$2" n="$3"
  local pids=() i start end elapsed total
  start=$(date +%s.%N)
  for ((i = 0; i < n; i++)); do
    one_stream "$dir/.io-probe-$i" "$mode" >/dev/null &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  end=$(date +%s.%N)
  # Aggregate rather than per-stream: total bytes moved over the wall clock of the
  # whole batch, which is the number a parallel scan actually experiences.
  total=$(( size_mb * n ))
  awk -v t="$total" -v s="$start" -v e="$end" -v n="$n" \
    'BEGIN { d = e - s; printf "  %2d stream(s)  %8.0f MB in %6.1fs = %7.0f MB/s (%6.0f MB/s each)\n",
             n, t, d, t / d, t / d / n }'
}

for path in "${paths[@]}"; do
  echo "=== $path"
  mkdir -p "$path" || { echo "  cannot create $path" >&2; continue; }
  topology "$path"
  need=$(( size_mb * max_streams ))
  free_mb=$(df -BM --output=avail "$path" | tail -1 | tr -dc '0-9')
  if (( free_mb < need + 1024 )); then
    echo "  skipping: needs ${need}M, has ${free_mb}M free" >&2
    continue
  fi

  echo "  -- sequential write"
  for n in $streams; do probe "$path" write "$n"; done
  echo "  -- sequential read (O_DIRECT, cache bypassed)"
  for n in $streams; do probe "$path" read "$n"; done

  if [[ "$keep" == 0 ]]; then rm -f "$path"/.io-probe-*; fi
  echo
done
