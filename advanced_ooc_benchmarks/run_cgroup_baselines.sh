#!/usr/bin/env bash
# Execute this standalone advanced OOC plan in independent user-systemd scopes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plan="${BENCHMARK_PLAN:-$here/benchmark-plan.yaml}"
[[ -f "$plan" ]] || { echo "Missing benchmark plan: $plan" >&2; exit 2; }
python="${BENCHMARK_PLAN_PYTHON:-$(awk -F: '/^[[:space:]]{2}python:[[:space:]]*/ {sub(/^[[:space:]]+/, "", $2); gsub(/[[:space:]]+$/, "", $2); print $2; exit}' "$plan")}"
: "${python:=python3}"
exec "$python" "$here/benchmark_plan.py" "$plan"
