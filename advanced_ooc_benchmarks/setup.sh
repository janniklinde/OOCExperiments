#!/usr/bin/env bash
# Prepare a host to execute benchmark-plan.yaml: install the baseline Python dependencies into
# the interpreter the plan configures as tools.python, then report on everything else the plan
# needs. Re-running is safe; every step is idempotent.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plan="${BENCHMARK_PLAN:-$here/benchmark-plan.yaml}"
[[ -f "$plan" ]] || { echo "Missing benchmark plan: $plan" >&2; exit 2; }

skip_systemds_python=0
for argument in "$@"; do
  case "$argument" in
    --skip-systemds-python) skip_systemds_python=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./setup.sh [--skip-systemds-python]

Installs requirements-baselines.txt into the plan's tools.python, creates that interpreter as a
virtualenv when it does not exist yet, installs the SystemDS Python bindings from the source tree
next to tools.systemds_jar, and verifies the remaining tools the plan needs.

Environment: BENCHMARK_PLAN overrides the plan path, SETUP_BOOTSTRAP_PYTHON the interpreter used
to create a missing virtualenv (default python3).
USAGE
      exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

# Read one scalar out of the plan, addressed as a top-level key or as block.key. This is a
# deliberate two-level subset of YAML rather than a parser: it must work before any interpreter
# with PyYAML exists, which is the situation setup.sh is here to fix.
plan_value() {
  awk -v path="$1" '
    BEGIN { depth = split(path, parts, ".") }
    { line = $0; sub(/[[:space:]]+#.*$/, "", line); sub(/[[:space:]]+$/, "", line) }
    line ~ /^[^[:space:]#]/ {
      key = line; sub(/:.*/, "", key); block = key
      if (depth == 1 && key == parts[1]) { sub(/^[^:]*:[[:space:]]*/, "", line); print line; exit }
      next
    }
    depth == 2 && block == parts[1] && line ~ "^  "parts[2]":" {
      sub(/^[^:]*:[[:space:]]*/, "", line); print line; exit
    }
  ' "$plan"
}

root="$(plan_value root)"
root="${root:-$(dirname "$here")/benchmark-data}"
root="${root//'${plan.dir}'/$here}"
expand() { local value="${1//'${plan.dir}'/$here}"; echo "${value//'${plan.root}'/$root}"; }

python="$(expand "$(plan_value tools.python)")"
: "${python:=python3}"
java="$(expand "$(plan_value tools.java)")"
spark_submit="$(expand "$(plan_value tools.spark_submit)")"
systemds_jar="$(expand "$(plan_value tools.systemds_jar)")"

status=0
note() { printf '%-6s %s\n' "$1" "$2"; }
fail() { note FAIL "$1"; status=1; }

echo "== Plan"
note plan "$plan"
note root "$root"
note python "$python"

echo
echo "== Python environment"
if [[ ! -x "$python" ]] && ! command -v "$python" >/dev/null 2>&1; then
  # tools.python usually names a virtualenv that simply has not been created on this host yet.
  venv="${python%/bin/python*}"
  if [[ "$venv" != "$python" ]]; then
    bootstrap="${SETUP_BOOTSTRAP_PYTHON:-python3}"
    command -v "$bootstrap" >/dev/null 2>&1 || { fail "no interpreter to create $venv (set SETUP_BOOTSTRAP_PYTHON)"; exit 1; }
    note create "$venv (via $bootstrap)"
    "$bootstrap" -m venv "$venv"
  else
    fail "configured interpreter does not exist and is not a virtualenv path: $python"
    exit 1
  fi
fi
note version "$("$python" --version 2>&1)"

echo
echo "== Baseline requirements"
"$python" -m pip install --disable-pip-version-check --quiet --upgrade pip
"$python" -m pip install --disable-pip-version-check --quiet -r "$here/requirements-baselines.txt"
note ok "requirements-baselines.txt"

echo
echo "== SystemDS Python bindings"
if [[ "$skip_systemds_python" == 1 ]]; then
  note skip "--skip-systemds-python"
elif "$python" -c "import systemds" >/dev/null 2>&1; then
  note ok "already installed"
else
  # The bindings are not released alongside this plan; take them from the checkout that produced
  # the configured JAR so the dataset preparation scripts match the engine under test.
  bindings="$(dirname "$(dirname "$systemds_jar")")/src/main/python"
  if [[ -f "$bindings/setup.py" ]]; then
    note install "$bindings"
    "$python" -m pip install --disable-pip-version-check --quiet --editable "$bindings"
  else
    fail "no bindings at $bindings; dataset preparation needs 'import systemds'"
  fi
fi

echo
echo "== Module check"
# Every module any implementation declares, enabled or not, so that flipping a run on later does
# not send you back here. yaml drives benchmark_plan.py itself; matplotlib drives the plots.
modules="$( { grep -o 'required_python_modules:[[:space:]]*\[[^]]*\]' "$plan" \
    | sed 's/.*\[//; s/\]//; s/,/ /g' | tr ' ' '\n'; printf 'yaml\nmatplotlib\n'; } \
  | sed '/^$/d' | sort -u | tr '\n' ' ')"
for module in $modules; do
  if "$python" -c "import $module" >/dev/null 2>&1; then note ok "$module"; else fail "$module"; fi
done

echo
echo "== External tools"
for entry in "java:$java" "spark_submit:$spark_submit"; do
  name="${entry%%:*}"; value="${entry#*:}"
  if [[ -n "$value" ]] && command -v "$value" >/dev/null 2>&1; then note ok "$name -> $value"
  else note WARN "$name is not executable: ${value:-<unset>}"; fi
done
if [[ -f "$systemds_jar" ]]; then note ok "systemds_jar -> $systemds_jar"
else note WARN "systemds_jar does not exist: ${systemds_jar:-<unset>} (build it with mvn package)"; fi
if command -v systemd-run >/dev/null 2>&1 && systemctl --user status >/dev/null 2>&1; then
  note ok "user systemd (systemd-run, systemctl --user)"
else
  note WARN "no working user systemd instance; ./run_cgroup_baselines.sh needs one"
fi
if [[ -d "$root" || -w "$(dirname "$root")" ]]; then note ok "root is writable: $root"
else note WARN "root is not writable: $root"; fi

echo
echo "== Plan validation"
if "$python" "$here/benchmark_plan.py" "$plan" --validate; then :; else fail "plan validation"; fi

echo
if [[ "$status" == 0 ]]; then echo "Setup complete. Run ./run_cgroup_baselines.sh"; else
  echo "Setup incomplete; see FAIL lines above." >&2; fi
exit "$status"
