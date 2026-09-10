#!/usr/bin/env bash
# Interactive one-time setup for the containerized benchmark runner.
#
# Asks for the directories compose needs (data root, SystemDS jar/lib/bindings),
# validates them, writes docker/.env, then builds the image and runs the preflight
# and the memory-cap gate. Afterwards `./bench.sh run|run-detached` just works.
#
# Re-running is safe: every question shows the previous answer as its default.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
env_file="$here/.env"
container="advanced-ooc-benchmarks"

[[ -t 0 ]] || { echo "error: this script is interactive; run it in a terminal" >&2; exit 1; }

note() { printf '%-6s %s\n' "$1" "$2"; }
fail() { note FAIL "$*"; exit 1; }

# Absolute path even when the file does not exist yet, as long as its parent does.
abspath() {
  local p="$1" d
  if [[ -d "$p" ]]; then (cd "$p" && pwd -P); return; fi
  d="$(dirname "$p")"
  if [[ -d "$d" ]]; then (cd "$d" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$p")"); else printf '%s\n' "$p"; fi
}

# ask VAR "prompt" "default" -- echoes the answer into VAR (empty input accepts the
# default, which is shown in brackets; deliberately not `read -e -i`, whose
# pre-filled buffer makes a typed path append to the default instead of replace it).
ask() {
  local __var="$1" __prompt="$2" __default="$3" __answer=""
  read -r -p "$__prompt [$__default]: " __answer </dev/tty || exit 1
  printf -v "$__var" '%s' "${__answer:-$__default}"
}

# ask_path VAR "prompt" "default" "description" -- loops until the path passes its check.
ask_path() {
  local __var="$1" __prompt="$2" __default="$3" __desc="$4" __check="$5" __value
  while true; do
    ask __value "$__prompt" "$__default"
    [[ -z "$__value" ]] && continue
    [[ "$__value" == *" "* ]] && { note WARN "spaces in paths break compose .env; pick another"; continue; }
    if "$__check" "$__value"; then printf -v "$__var" '%s' "$(abspath "$__value")"; return; fi
    note WARN "$__value: not a usable $__desc"
  done
}

jar_ok()     { [[ -s "$1" ]]; }
lib_ok()     { [[ -d "$1" ]] && ls "$1"/*.jar >/dev/null 2>&1; }
python_ok()  { [[ -f "$1/setup.py" ]]; }
data_ok()    { [[ -d "$1" ]] || mkdir -p "$1" 2>/dev/null; }
data_writable() { [[ -d "$1" && -w "$1" ]]; }

echo "== Containerized OOC benchmark setup"
note repo "$repo_root"
note uid:gid "$(id -u):$(id -g)  (bind-mount ownership; edit .env by hand to override)"

# Defaults from the previous .env when one exists, else the sibling-checkout layout.
prev() {
  [[ -f "$env_file" ]] || return 0
  sed -n "s/^$1=//p" "$env_file" 2>/dev/null | head -1 || true
}
def_jar="$(prev SYSTEMDS_JAR)";      [[ -n "$def_jar" ]] || def_jar="$repo_root/../systemds/target/SystemDS.jar"
def_lib="$(prev SYSTEMDS_LIB_DIR)";  [[ -n "$def_lib" ]] || def_lib="$(dirname "$def_jar")/lib"
def_py="$(prev SYSTEMDS_PYTHON_DIR)";[[ -n "$def_py" ]] || def_py="$repo_root/../systemds/src/main/python"
def_data="$(prev BENCH_DATA_DIR)";   [[ -n "$def_data" ]] || def_data="$here/data"
def_spark="$(prev INSTALL_SPARK)";   [[ -n "$def_spark" ]] || def_spark="true"

echo
echo "== Paths (Enter accepts the [default])"
ask_path systemds_jar "SystemDS jar (prebuilt; mounted at /opt/systemds/SystemDS.jar)" "$def_jar" "jar file" jar_ok
ask_path systemds_lib "jar dependency dir (its manifest Class-Path resolves lib/...)" "$def_lib" "directory of jars" lib_ok
ask_path systemds_py "SystemDS Python bindings (src/main/python of the same checkout)" "$def_py" "bindings directory" python_ok
ask_path data_dir "data + results root, on a real disk (mounted at /bench/data)" "$def_data" "data root" data_ok
data_writable "$data_dir" || fail "$data_dir is not writable"

ask spark "install Spark into the image? (true/false)" "$def_spark"
[[ "$spark" == true || "$spark" == false ]] || fail "INSTALL_SPARK must be true or false"
if [[ "$spark" == false ]]; then
  note WARN "the shipped plan's systemds-spark template aborts the preflight without spark-submit"
fi

free_mb="$(df -BM --output=avail "$data_dir" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if (( free_mb > 0 && free_mb < 358400 )); then
  note WARN "only $((free_mb / 1024)) GiB free on $data_dir; the full dense sweep wants ~350 GiB (subsets via --only fit less)"
else
  note ok "$((free_mb / 1024)) GiB free on $data_dir"
fi

echo
echo "== Summary"
note jar "$systemds_jar ($(du -h "$systemds_jar" | cut -f1))"
note lib "$systemds_lib ($(ls "$systemds_lib"/*.jar 2>/dev/null | wc -l) jars)"
note python "$systemds_py"
note data "$data_dir"
note spark "$spark"
if [[ -f "$env_file" ]]; then
  cp "$env_file" "$env_file.bak.$(date +%Y%m%dT%H%M%S)"
  note backup "existing .env saved beside the new one"
fi
ask go "write $env_file and build + preflight now?" "Y"
[[ "$go" =~ ^([Yy]([Ee][Ss])?)?$ ]] || { note ok "nothing written; re-run any time"; exit 0; }

cat > "$env_file" <<EOF
# Written by setup-interactive.sh on $(date -Is)
BENCH_UID=$(id -u)
BENCH_GID=$(id -g)
REPO_DIR=$repo_root
BENCH_DATA_DIR=$data_dir
SYSTEMDS_JAR=$systemds_jar
SYSTEMDS_LIB_DIR=$systemds_lib
SYSTEMDS_PYTHON_DIR=$systemds_py
INSTALL_SPARK=$spark
PYSPARK_VERSION=3.5.3
JAVA_VERSION=21
SHM_SIZE=2gb
EOF
note ok "wrote $env_file"

echo
echo "== Docker access"
docker_prefix=()
if ! docker ps >/dev/null 2>&1; then
  if sudo -n docker ps >/dev/null 2>&1; then
    docker_prefix=(sudo)
    note ok "docker via sudo (passwordless); exporting BENCH_DOCKER for bench.sh"
  else
    fail "docker is unusable as $(id -un): join the docker group (root-equivalent) or install docker"
  fi
else
  note ok "docker works without sudo"
fi
dc() { if [[ ${#docker_prefix[@]} -gt 0 ]]; then sudo docker "$@"; else docker "$@"; fi; }
bench() {
  if [[ ${#docker_prefix[@]} -gt 0 ]]; then BENCH_DOCKER="sudo docker" "$here/bench.sh" "$@"
  else "$here/bench.sh" "$@"; fi
}
dc compose version >/dev/null 2>&1 || dc version >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1 ||
  fail "no working compose; install the plugin into ~/.docker/cli-plugins"

echo
echo "== Storage probe (O_DIRECT, ~5 GiB, removed afterwards)"
ask probe "run a quick io_probe.sh on the data root?" "Y"
if [[ "$probe" =~ ^([Yy]([Ee][Ss])?)?$ ]]; then
  IO_PROBE_SIZE_MB=1024 IO_PROBE_STREAMS="1 4" "$here/../io_probe.sh" "$data_dir" || note WARN "probe failed; see io_probe.sh --help"
  echo "  (full-scale probe later: IO_PROBE_SIZE_MB=4096 IO_PROBE_STREAMS='1 4 16' ./io_probe.sh $data_dir)"
fi

echo
echo "== Build"
bench build || fail "image build failed"

echo
echo "== Preflight (starts the container)"
bench preflight || note WARN "preflight reported problems; fix FAIL lines before running anything"

echo
echo "== Memory-cap gate (the one check that cannot be trusted to preflight)"
uid="$(id -u)"
gate_out="$(dc exec -u "$uid" -e XDG_RUNTIME_DIR=/run/user/$uid "$container" \
  systemd-run --user --scope -q -p MemoryMax=64M -p MemorySwapMax=0 \
  python3 -c "b=bytearray(256*1024*1024); print('ALLOCATED')" 2>&1 || true)"
if grep -q ALLOCATED <<<"$gate_out"; then
  note FAIL "the cap is FICTION: the allocation printed instead of being killed"
  note FAIL "every memory-axis number would be meaningless; fix controller delegation first"
  exit 1
fi
note ok "memory cap is enforced (allocation killed inside the scope)"

echo
echo "== Ready. Next steps"
cat <<EOF
  one workload first:  ./bench.sh run /workspace/advanced_ooc_benchmarks/benchmark-plan.yaml --only <run-id>
  full sweep:         ./bench.sh run-detached        (survives closing the terminal)
  follow / status:    ./bench.sh tail | ./bench.sh status
  stop:               ./bench.sh down
EOF
