#!/usr/bin/env bash
# Pull benchmark results back from the remote host that ran docker/bench.sh.
#
# The remote data root holds two things side by side under bench-results/:
# `run-<stamp>.log`, one per detached sweep, and `<invocation-id>/`, the tree
# benchmark_plan.py writes for that sweep (results.csv per run, per-case logs,
# metrics, telemetry, and the resolved plan). By default this fetches the newest
# invocation and its driver log; --all mirrors the whole bench-results directory.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

host="${BENCH_REMOTE_HOST:-so014}"
data="${BENCH_REMOTE_DATA_DIR:-/home/lindemann/bench/data}"
dest="${BENCH_RESULTS_DIR:-$here/results-remote}"
invocation=""
fetch_all=0
with_telemetry=1
list_only=0
dry=()

usage() {
  cat <<'USAGE'
Usage: ./download_results_remote.sh [options]

  --host HOST         ssh destination (default: so014, or $BENCH_REMOTE_HOST)
  --data-dir PATH     remote data root (default: /home/lindemann/bench/data,
                      or $BENCH_REMOTE_DATA_DIR)
  --dest PATH         local destination (default: ./results-remote,
                      or $BENCH_RESULTS_DIR)
  --invocation ID     fetch this invocation instead of the newest one
  --all               mirror every invocation and driver log
  --no-telemetry      skip the per-case *.telemetry.csv traces (the bulk of it)
  --list              show what is on the remote and exit
  --dry-run           show what rsync would transfer, change nothing
  -h, --help          this message

Nothing is deleted locally: each invocation lands in its own directory, so
repeated runs accumulate and a re-fetch of an in-progress sweep just tops it up.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="$2"; shift 2 ;;
    --data-dir) data="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    --invocation) invocation="$2"; shift 2 ;;
    --all) fetch_all=1; shift ;;
    --no-telemetry) with_telemetry=0; shift ;;
    --list) list_only=1; shift ;;
    --dry-run) dry=(--dry-run); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 2; }
data="${data%/}"
results="$data/bench-results"

# One ssh round trip for the listing: an invocation directory is a plain
# timestamp directory, while the driver logs are files named run-<stamp>.log.
listing="$(ssh "$host" "cd '$results' 2>/dev/null || { echo '__missing__'; exit 0; }
  find . -maxdepth 1 -mindepth 1 -type d -printf 'dir %T@ %f\n'
  find . -maxdepth 1 -mindepth 1 -type f -name 'run-*.log' -printf 'log %T@ %f\n'")"

if [[ "$listing" == *__missing__* ]]; then
  echo "no $results on $host; has a sweep been started?" >&2
  exit 2
fi

invocations="$(awk '$1 == "dir" {print $2, $3}' <<< "$listing" | sort -rn | cut -d' ' -f2-)"
logs="$(awk '$1 == "log" {print $2, $3}' <<< "$listing" | sort -rn | cut -d' ' -f2-)"

if [[ "$list_only" == 1 ]]; then
  echo "host        $host"
  echo "results     $results"
  echo
  echo "invocations (newest first)"
  if [[ -n "$invocations" ]]; then sed 's/^/  /' <<< "$invocations"; else echo "  none"; fi
  echo
  echo "driver logs (newest first)"
  if [[ -n "$logs" ]]; then sed 's/^/  /' <<< "$logs"; else echo "  none"; fi
  exit 0
fi

exclude=()
[[ "$with_telemetry" == 0 ]] && exclude=(--exclude='*.telemetry.csv')

mkdir -p "$dest"

if [[ "$fetch_all" == 1 ]]; then
  [[ -n "$invocation" ]] && { echo "--all and --invocation are mutually exclusive" >&2; exit 2; }
  echo "host        $host"
  echo "results     $results"
  echo "dest        $dest"
  echo
  echo "== everything under bench-results"
  rsync -az "${dry[@]}" "${exclude[@]}" --info=stats1 --partial \
    "$host:$results/" "$dest/"
  exit 0
fi

if [[ -z "$invocation" ]]; then
  invocation="$(head -1 <<< "$invocations")"
  [[ -n "$invocation" ]] || { echo "no invocation directories in $results" >&2; exit 2; }
else
  grep -qxF "$invocation" <<< "$invocations" ||
    { echo "no such invocation on $host: $invocation" >&2
      echo "run with --list to see what is there" >&2; exit 2; }
fi

echo "host        $host"
echo "results     $results"
echo "invocation  $invocation"
echo "dest        $dest/$invocation"

echo
echo "== invocation tree"
# --partial so an interrupted pull of a large telemetry trace resumes rather
# than restarting, and no --delete: a sweep still in progress keeps growing.
rsync -az "${dry[@]}" "${exclude[@]}" --info=stats1 --partial \
  "$host:$results/$invocation/" "$dest/$invocation/"

if [[ -n "$logs" ]]; then
  newest_log="$(head -1 <<< "$logs")"
  echo
  echo "== driver log $newest_log"
  rsync -az "${dry[@]}" --info=stats1 --partial \
    "$host:$results/$newest_log" "$dest/$newest_log"
fi

if [[ ${#dry[@]} -gt 0 ]]; then
  echo
  echo "dry run; nothing was written"
  exit 0
fi

echo
echo "runs with results:"
found=0
for csv in "$dest/$invocation"/*/results.csv; do
  [[ -f "$csv" ]] || continue
  found=1
  # Header plus one row per implementation x rep; count only completed rows.
  rows=$(( $(wc -l < "$csv") - 1 ))
  ok=$(awk -F, 'NR > 1 && $5 == "ok"' "$csv" | wc -l)
  printf '  %-32s %3d rows, %3d ok\n' "$(basename "$(dirname "$csv")")" "$rows" "$ok"
done
[[ "$found" == 1 ]] || echo "  none yet (the sweep may still be preparing datasets)"
