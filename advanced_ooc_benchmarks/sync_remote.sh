#!/usr/bin/env bash
# Push this benchmark directory and the SystemDS build it names to a remote host
# that runs the suite through docker/bench.sh.
#
# Three things travel: the plan directory itself, the jar from the plan's
# `tools.systemds_jar`, and that jar's sibling `lib/` -- the manifest carries
# `Class-Path: lib/...` resolved relative to the jar, so the jar alone cannot
# start. Nothing else does: the container builds its own Python environment and
# its own spark-submit, and `make_container_plan.py` rewrites the plan's
# host-specific paths at launch, so the plan needs no editing for the remote.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plan="${BENCHMARK_PLAN:-$here/benchmark-plan.yaml}"

host="${BENCH_REMOTE_HOST:-so014}"
remote="${BENCH_REMOTE_DIR:-bench}"
data_dir="${BENCH_REMOTE_DATA_DIR:-}"
write_env=1
force_env=0
send_jar=1
dry=()

usage() {
  cat <<'USAGE'
Usage: ./sync_remote.sh [HOST] [REMOTE_DIR] [options]

  HOST         ssh destination (default: so014, or $BENCH_REMOTE_HOST)
  REMOTE_DIR   path under the remote home (default: bench, or $BENCH_REMOTE_DIR)

  --data-dir PATH   remote dataset/results root written into docker/.env
                    (default: <remote home>/<REMOTE_DIR>/data)
  --no-jar          skip the SystemDS jar and lib/ (~350 MB); the Python
                    bindings are sent regardless, being small and required
  --no-env          do not write docker/.env on the remote
  --force-env       overwrite an existing remote docker/.env
  --dry-run         show what rsync would transfer, change nothing
  -h, --help        this message
USAGE
}

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) data_dir="$2"; shift 2 ;;
    --no-jar) send_jar=0; shift ;;
    --no-env) write_env=0; shift ;;
    --force-env) force_env=1; shift ;;
    --dry-run) dry=(--dry-run); shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done
[[ ${#positional[@]} -ge 1 ]] && host="${positional[0]}"
[[ ${#positional[@]} -ge 2 ]] && remote="${positional[1]}"
[[ ${#positional[@]} -gt 2 ]] && { echo "too many arguments" >&2; usage >&2; exit 2; }
remote="${remote#\~/}"; remote="${remote%/}"

[[ -f "$plan" ]] || { echo "missing benchmark plan: $plan" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 2; }

# Same two-level YAML subset that setup.sh reads, for the same reason: this must
# work without assuming an interpreter with PyYAML on the machine running it.
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

jar="$(plan_value tools.systemds_jar)"
jar="${jar/#\~/$HOME}"
lib="$(dirname "$jar")/lib"
# The Python bindings ship with the jar rather than separately: dataset preparation
# talks to this exact engine through them, so a binding from a different checkout is
# a version skew waiting to happen.
bindings="$(dirname "$(dirname "$jar")")/src/main/python"

echo "host        $host"
echo "remote dir  ~/$remote"
echo "plan        $plan"
if [[ "$send_jar" == 1 ]]; then
  [[ -f "$jar" ]] || { echo "tools.systemds_jar does not exist: $jar" >&2; exit 2; }
  echo "jar         $jar ($(du -h "$jar" | cut -f1))"
  if [[ -d "$lib" ]]; then
    echo "lib         $lib ($(du -sh "$lib" | cut -f1), $(find "$lib" -name '*.jar' | wc -l) jars)"
  else
    echo "lib         none beside the jar; skipping" >&2
    lib=""
  fi
fi
if [[ -f "$bindings/setup.py" ]]; then
  echo "bindings    $bindings ($(du -sh "$bindings" | cut -f1))"
else
  echo "tools.systemds_jar has no src/main/python beside it: $bindings" >&2
  echo "dataset preparation imports systemds and will fail without it" >&2
  exit 2
fi

ssh "$host" "mkdir -p ~/$remote/advanced_ooc_benchmarks ~/$remote/systemds"

echo
echo "== benchmark directory"
# --delete keeps the remote a mirror; results/ and caches stay local because a
# remote run regenerates its own under the plan's root.
rsync -az --delete "${dry[@]}" --info=stats1 \
  --exclude='*/results' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='scratch_space' --exclude='.env' --exclude='*.container.yaml' \
  --exclude='docker/data' \
  "$here/" "$host:$remote/advanced_ooc_benchmarks/"

if [[ "$send_jar" == 1 ]]; then
  echo
  echo "== SystemDS build"
  # The jar and lib/ must stay siblings on the remote for the manifest Class-Path
  # to resolve, which is why they go to one directory rather than two.
  rsync -az "${dry[@]}" --info=stats1 "$jar" "$host:$remote/systemds/SystemDS.jar"
  if [[ -n "$lib" ]]; then
    rsync -az --delete "${dry[@]}" --info=stats1 "$lib/" "$host:$remote/systemds/lib/"
  fi
fi

echo
echo "== SystemDS Python bindings"
# Always, even with --no-jar: five megabytes against the 350 MB build, and the
# container refuses to start dataset preparation without them.
# Excluding build metadata, because the container installs from this tree and a
# local egg-info would shadow the version it builds there.
rsync -az --delete "${dry[@]}" --info=stats1 \
  --exclude='__pycache__' --exclude='*.egg-info' --exclude='build' --exclude='dist' \
  "$bindings/" "$host:$remote/systemds/python/"

if [[ ${#dry[@]} -gt 0 ]]; then
  echo
  echo "dry run; nothing was written"
  exit 0
fi

if [[ "$write_env" == 1 ]]; then
  echo
  echo "== docker/.env"
  ssh "$host" \
    "REMOTE='$remote' DATA_DIR='$data_dir' FORCE='$force_env' bash -s" <<'REMOTE_ENV'
set -euo pipefail
env_file="$HOME/$REMOTE/advanced_ooc_benchmarks/docker/.env"
data="${DATA_DIR:-$HOME/$REMOTE/data}"
if [[ -f "$env_file" && "$FORCE" != 1 ]]; then
  echo "keeping existing $env_file (--force-env to replace)"
else
  cat > "$env_file" <<EOF
BENCH_UID=$(id -u)
BENCH_GID=$(id -g)
REPO_DIR=$HOME/$REMOTE
BENCH_DATA_DIR=$data
SYSTEMDS_JAR=$HOME/$REMOTE/systemds/SystemDS.jar
SYSTEMDS_LIB_DIR=$HOME/$REMOTE/systemds/lib
SYSTEMDS_PYTHON_DIR=$HOME/$REMOTE/systemds/python
INSTALL_SPARK=true
PYSPARK_VERSION=3.5.3
JAVA_VERSION=21
SHM_SIZE=2gb
EOF
  echo "wrote $env_file"
fi
# A warning, not a failure: the sync itself has already succeeded, and a data
# root that needs root or a mount is the operator's to create.
mkdir -p "$data" 2>/dev/null || echo "warning: could not create $data; create it before running" >&2
free=$(df -BG --output=avail "$data" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")
if [[ -n "$free" && "$free" -lt 350 ]]; then
  echo "warning: $data has ${free}G free; the enabled sweep needs roughly 350G" >&2
else
  echo "data root $data (${free:-?}G free)"
fi
command -v docker >/dev/null || echo "warning: no docker on this host" >&2
REMOTE_ENV
fi

cat <<NEXT

Synced. On $host:

  cd ~/$remote/advanced_ooc_benchmarks/docker
  ./bench.sh build
  ./bench.sh run
NEXT
