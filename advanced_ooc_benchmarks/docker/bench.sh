#!/usr/bin/env bash
# Host-side wrapper around the benchmark container.
#
#   ./bench.sh build            build the image
#   ./bench.sh up               start the container (systemd + user manager)
#   ./bench.sh preflight        check cgroups, systemd, io.stat, tools, disk
#   ./bench.sh run [PLAN]       run run_cgroup_baselines.sh inside the container
#   ./bench.sh run-detached     same, in the background, logging to the data root
#   ./bench.sh tail [LINES]     follow the newest detached run log
#   ./bench.sh status           is a sweep in progress?
#   ./bench.sh shell            interactive shell as the benchmark user
#   ./bench.sh logs             follow container logs
#   ./bench.sh down             stop and remove the container
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
service=bench
container="${BENCH_CONTAINER_NAME:-advanced-ooc-benchmarks}"

# Compose V2 as a docker plugin, or the standalone V1 binary. Set BENCH_COMPOSE to
# override (e.g. "sudo docker compose").
if [[ -n "${BENCH_COMPOSE:-}" ]]; then
  read -r -a compose <<< "$BENCH_COMPOSE"
elif docker compose version >/dev/null 2>&1; then
  compose=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose=(docker-compose)
else
  echo "error: neither 'docker compose' nor 'docker-compose' works here." >&2
  echo "       docker: $(command -v docker || echo 'not found')" >&2
  echo "       set BENCH_COMPOSE if it lives somewhere else." >&2
  exit 2
fi

# Run from the compose file's directory rather than passing --project-directory:
# that flag is missing from older CLIs, and the working directory has the same
# effect on the project name, the .env lookup, and relative bind-mount paths.
run_compose() { (cd "$here" && "${compose[@]}" "$@"); }

uid="${BENCH_UID:-$(id -u)}"
user_env=(-e "XDG_RUNTIME_DIR=/run/user/${uid}"
          -e "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${uid}/bus")
entry=/workspace/advanced_ooc_benchmarks/docker/run-in-container.sh
log_dir=/bench/data/bench-results

# Only ask for a TTY when there is one, so `nohup ./bench.sh run &` and any other
# non-interactive caller work rather than failing with "the input device is not a TTY".
tty_flags=()
[[ -t 0 && -t 1 ]] && tty_flags=(-it)

start() {
  run_compose up -d "$service"
  # systemd needs a moment to reach the user manager; linger starts it at boot,
  # and starting it explicitly is idempotent if logind has not yet done so.
  for _ in $(seq 60); do
    if docker exec "$container" systemctl is-system-running --wait >/dev/null 2>&1 ||
       docker exec "$container" systemctl is-system-running 2>/dev/null |
         grep -qE 'running|degraded'; then
      break
    fi
    sleep 1
  done
  docker exec -u root "$container" systemctl start "user@${uid}.service"
  for _ in $(seq 30); do
    if docker exec -u "$uid" "${user_env[@]}" "$container" systemctl --user status >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "error: the user systemd manager did not come up; check './bench.sh logs'" >&2
  exit 1
}

case "${1:-}" in
  build) shift; run_compose build "$@" ;;
  up)    start; echo "container '$container' is ready" ;;
  run)
    shift
    start
    docker exec "${tty_flags[@]}" -u "$uid" "${user_env[@]}" "$container" "$entry" "$@"
    ;;
  run-detached)
    # The exec is owned by the docker daemon, so the sweep survives the ssh session
    # that started it; no nohup, no terminal to keep open. The log goes to the data
    # root because that is bind-mounted, so it is readable from the host while the
    # run is still going.
    shift
    start
    stamp="$(date +%Y%m%dT%H%M%S)"
    log="$log_dir/run-$stamp.log"
    quoted="$entry"
    for argument in "$@"; do quoted+=" $(printf '%q' "$argument")"; done
    docker exec -d -u "$uid" "${user_env[@]}" "$container" \
      bash -c "mkdir -p $log_dir && exec $quoted >> $log 2>&1"
    echo "started in the background; log: \$BENCH_DATA_DIR/bench-results/run-$stamp.log"
    echo "follow it with: $0 tail"
    ;;
  tail)
    shift
    docker exec "${tty_flags[@]}" -u "$uid" "$container" \
      bash -c "tail -n \${1:-40} -f \$(ls -t $log_dir/run-*.log | head -1)" "$@"
    ;;
  preflight)
    start
    docker exec "${tty_flags[@]}" -u "$uid" "${user_env[@]}" "$container" \
      /workspace/advanced_ooc_benchmarks/docker/preflight.sh
    ;;
  status)
    if docker exec -u "$uid" "$container" pgrep -af benchmark_plan.py >/dev/null 2>&1; then
      docker exec -u "$uid" "$container" pgrep -af benchmark_plan.py
    else
      echo "no benchmark run in progress"
    fi
    ;;
  shell)
    start
    docker exec -it -u "$uid" "${user_env[@]}" -w /workspace/advanced_ooc_benchmarks \
      "$container" bash -l
    ;;
  logs) shift; run_compose logs -f "$@" ;;
  down) shift; run_compose down "$@" ;;
  *) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
