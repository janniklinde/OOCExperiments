#!/usr/bin/env bash
# Container-side entry point: rewrite the plan for the container layout, verify
# the preconditions the runner cannot recover from, then hand over to
# run_cgroup_baselines.sh. Runs as the unprivileged benchmark user.
#
# Usage: run-in-container.sh [SOURCE_PLAN]
set -euo pipefail

plan_dir="${BENCH_PLAN_DIR:-/workspace/advanced_ooc_benchmarks}"
source_plan="${1:-${BENCHMARK_SOURCE_PLAN:-$plan_dir/benchmark-plan.yaml}}"
container_plan="${BENCHMARK_PLAN:-$plan_dir/benchmark-plan.container.yaml}"
python="${BENCH_CONTAINER_PYTHON:-/opt/bench-venv/bin/python}"
jar="${BENCH_CONTAINER_JAR:-/opt/systemds/SystemDS.jar}"
data_root="${BENCH_CONTAINER_ROOT:-/bench/data}"

fail() { echo "error: $*" >&2; exit 1; }

[[ -f "$source_plan" ]] || fail "missing source plan $source_plan (is the repository mounted at /workspace?)"
[[ -x "$python" ]] || fail "missing container Python $python"
[[ -s "$jar" ]] || fail "missing SystemDS jar $jar (mount a prebuilt jar; see docker/README.md)"
[[ -d "$data_root" && -w "$data_root" ]] || fail "$data_root is not a writable directory"

# The runner refuses to start without a user systemd manager, and the failure
# mode is otherwise a bare RuntimeError halfway through the preflight.
systemctl --user status >/dev/null 2>&1 || fail \
  "no user systemd manager for $(id -un); start it with: docker exec -u root advanced-ooc-benchmarks systemctl start user@$(id -u).service"

# The io controller must reach the scope, or io.stat is empty and the runner
# records no byte-accurate I/O.
if ! grep -qw io "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers" 2>/dev/null; then
  echo "warning: the io controller is not delegated to the user manager; I/O byte metrics will be missing" >&2
fi

"$python" "$plan_dir/docker/make_container_plan.py" "$source_plan" -o "$container_plan"

# Largest profile in the plan versus what the container may actually use.
limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [[ "$limit" != max ]]; then
  echo "note: this container is itself capped at $((limit >> 30)) GiB; plan profiles above that will OOM" >&2
fi

export BENCHMARK_PLAN="$container_plan"
export BENCHMARK_PLAN_PYTHON="$python"
exec "$plan_dir/run_cgroup_baselines.sh"
