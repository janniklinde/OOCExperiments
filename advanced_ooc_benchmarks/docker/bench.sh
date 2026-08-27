#!/usr/bin/env bash
# Host-side wrapper around the benchmark container.
#
#   ./bench.sh build            build the image
#   ./bench.sh up               start the container (systemd + user manager)
#   ./bench.sh run [PLAN]       run run_cgroup_baselines.sh inside the container
#   ./bench.sh shell            interactive shell as the benchmark user
#   ./bench.sh logs             follow container logs
#   ./bench.sh down             stop and remove the container
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
service=bench
container="${BENCH_CONTAINER_NAME:-advanced-ooc-benchmarks}"
compose=(docker compose --project-directory "$here")

uid="${BENCH_UID:-$(id -u)}"
user_env=(-e "XDG_RUNTIME_DIR=/run/user/${uid}"
          -e "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${uid}/bus")

start() {
  "${compose[@]}" up -d "$service"
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
  build) shift; "${compose[@]}" build "$@" ;;
  up)    start; echo "container '$container' is ready" ;;
  run)
    shift
    start
    docker exec -it -u "$uid" "${user_env[@]}" "$container" \
      /workspace/advanced_ooc_benchmarks/docker/run-in-container.sh "$@"
    ;;
  shell)
    start
    docker exec -it -u "$uid" "${user_env[@]}" -w /workspace/advanced_ooc_benchmarks \
      "$container" bash -l
    ;;
  logs) shift; "${compose[@]}" logs -f "$@" ;;
  down) shift; "${compose[@]}" down "$@" ;;
  *) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
