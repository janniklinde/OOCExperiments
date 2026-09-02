#!/usr/bin/env bash
#
# Regenerate the per-workload figures for the most recent benchmark invocation(s).
#
# By default only the newest invocation under BENCH_RESULTS_DIR is rendered, which covers
# exactly the workloads that run actually executed. Because a run often omits workloads
# (crash, interrupted plan, deliberate subset), DEPTH lets the script keep walking backwards:
# invocations are visited newest first, and any workload already rendered from a newer run is
# skipped in the older ones. The result is one up-to-date figure set per workload, each taken
# from the most recent invocation that contains it.
#
# Existing figures are always overwritten.
#
# Usage:  ./regenerate_figures.sh [BENCH_RESULTS_DIR] [DEPTH]
#         ./regenerate_figures.sh                       # newest invocation only
#         ./regenerate_figures.sh "" 5                  # newest 5, filling gaps backwards
#         ./regenerate_figures.sh ~/other-results 3
#         ./regenerate_figures.sh ~/other-results/20260829T142359.823028+0200
#
set -euo pipefail

DEFAULT_ROOT="/media/jannik/data/OOCExperiments/bench-results"
SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZER="$SUITE_DIR/visualize_invocation.py"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    exit 0
fi

ROOT="${1:-$DEFAULT_ROOT}"
ROOT="${ROOT:-$DEFAULT_ROOT}"          # an empty first argument also means "use the default"
DEPTH="${2:-1}"

if [[ ! -d "$ROOT" ]]; then
    echo "error: no such directory: $ROOT" >&2
    exit 1
fi
if [[ ! "$DEPTH" =~ ^[0-9]+$ ]] || (( DEPTH < 1 )); then
    echo "error: DEPTH must be a positive integer, got: $DEPTH" >&2
    exit 1
fi

# An invocation directory is identified by its expanded plan; accept one directly.
invocations=()
if [[ -f "$ROOT/expanded-plan.yaml" ]]; then
    invocations=("$ROOT")
    if (( DEPTH > 1 )); then
        echo "note: $ROOT is a single invocation, ignoring DEPTH=$DEPTH" >&2
    fi
else
    while IFS= read -r invocation; do
        invocations+=("$invocation")
    done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d \
                  -exec test -f '{}/expanded-plan.yaml' ';' -printf '%T@\t%p\n' \
             | sort -rn | cut -f2- | head -n "$DEPTH")
fi

if (( ${#invocations[@]} == 0 )); then
    echo "error: no benchmark invocation (a directory holding expanded-plan.yaml) under $ROOT" >&2
    exit 1
fi

rendered=()          # workloads already covered by a newer invocation
summary=()           # "<workload>\t<invocation>" lines for the closing report

for invocation in "${invocations[@]}"; do
    skip="$(IFS=,; echo "${rendered[*]:-}")"
    echo "==> $(basename "$invocation")${skip:+ (skipping: $skip)}"

    # Report the failure but keep walking back: an older invocation may still fill the gaps.
    if ! output="$("$PYTHON" "$VISUALIZER" "$invocation" --skip-workloads "$skip" 2>&1)"; then
        printf '%s\n' "$output" >&2
        echo "warning: rendering failed for $invocation, continuing with older invocations" >&2
        continue
    fi
    printf '%s\n' "$output" | grep -v '^RENDERED	' || true

    while IFS=$'\t' read -r _ workload _; do
        [[ -n "$workload" ]] || continue
        rendered+=("$workload")
        summary+=("$workload"$'\t'"$(basename "$invocation")")
    done < <(printf '%s\n' "$output" | grep '^RENDERED	' || true)
done

if (( ${#summary[@]} == 0 )); then
    echo "No workloads rendered." >&2
    exit 1
fi

echo
echo "Up-to-date figures (${#summary[@]} workloads, source invocation per workload):"
printf '%s\n' "${summary[@]}" | sort \
    | awk -F'\t' '{ printf "  %-22s %s\n", $1, $2 }'
