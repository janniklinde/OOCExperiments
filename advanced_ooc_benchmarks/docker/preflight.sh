#!/usr/bin/env bash
# Check, from inside the container, everything a run needs before committing hours
# to it. Reports; changes nothing. Run it with `../docker/bench.sh preflight`.
set -uo pipefail
status=0
note() { printf '%-6s %s\n' "$1" "$2"; }
fail() { note FAIL "$1"; status=1; }

python="${BENCH_CONTAINER_PYTHON:-/opt/bench-venv/bin/python}"
jar="${BENCH_CONTAINER_JAR:-/opt/systemds/SystemDS.jar}"
root="${BENCH_CONTAINER_ROOT:-/bench/data}"
plan_dir="${BENCH_PLAN_DIR:-/workspace/advanced_ooc_benchmarks}"
uid="$(id -u)"
user_cgroup="/sys/fs/cgroup/user.slice/user-$uid.slice/user@$uid.service"

echo "== cgroup v2"
if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
  note ok "unified hierarchy: $(cat /sys/fs/cgroup/cgroup.controllers)"
else
  fail "no cgroup-v2 unified hierarchy at /sys/fs/cgroup"
fi
if [[ -f "$user_cgroup/cgroup.controllers" ]]; then
  delegated="$(cat "$user_cgroup/cgroup.controllers")"
  note ok "delegated to user@$uid: $delegated"
  for controller in memory io cpu pids; do
    if grep -qw "$controller" <<< "$delegated"; then note ok "  $controller"
    elif [[ "$controller" == io ]]; then note WARN "  io missing: byte-accurate I/O will not be recorded"
    else fail "  $controller is not delegated"; fi
  done
else
  fail "no user@$uid.service cgroup; the user manager is not running"
fi

echo
echo "== systemd"
if command -v systemd-run >/dev/null && systemctl --user status >/dev/null 2>&1; then
  note ok "user manager reachable"
  if systemd-run --user --scope --collect --quiet -p MemoryMax=256M \
       -p MemoryAccounting=yes -p IOAccounting=yes true 2>/dev/null; then
    note ok "scope creation with MemoryMax/IOAccounting"
  else
    fail "systemd-run --user --scope failed; the runner cannot measure anything"
  fi
else
  fail "no working user systemd manager (systemctl --user status)"
fi

echo
echo "== io.stat"
probe="$root/.preflight-io-probe"
scope_io=""
if systemd-run --user --scope --collect --quiet --unit=ooc-preflight-io \
     -p IOAccounting=yes -- bash -c \
     "dd if=/dev/urandom of=$probe bs=1M count=32 oflag=direct status=none 2>/dev/null ||
      dd if=/dev/urandom of=$probe bs=1M count=32 conv=fsync status=none;
      cat /sys/fs/cgroup/\$(cat /proc/self/cgroup | cut -d: -f3)/io.stat" \
     > /tmp/preflight-io.txt 2>/dev/null; then
  scope_io="$(grep -c . /tmp/preflight-io.txt 2>/dev/null || echo 0)"
fi
rm -f "$probe" /tmp/preflight-io.txt
if [[ "${scope_io:-0}" -gt 0 ]]; then
  note ok "io.stat is populated inside a scope"
else
  note WARN "io.stat is empty; runs record no byte-accurate I/O (the runner falls back
       to GNU time's rusage block counters, which are coarser)"
fi

echo
echo "== page cache"
if [[ -w /proc/sys/vm/drop_caches ]]; then
  note ok "/proc/sys/vm/drop_caches is writable (cold caches between runs)"
else
  note WARN "/proc/sys/vm/drop_caches is not writable; drop_caches.py falls back to
       per-file POSIX_FADV_DONTNEED, which is less thorough"
fi

echo
echo "== tools"
[[ -x "$python" ]] && note ok "python $("$python" --version 2>&1)" || fail "missing $python"
[[ -s "$jar" ]] && note ok "jar $jar ($(du -h "$jar" | cut -f1))" || fail "missing or empty $jar"
if [[ -d /opt/systemds/lib ]] && [[ -n "$(ls -A /opt/systemds/lib 2>/dev/null)" ]]; then
  note ok "lib $(ls /opt/systemds/lib/*.jar 2>/dev/null | wc -l) jars"
else
  fail "/opt/systemds/lib is empty; the jar's manifest Class-Path cannot resolve"
fi
# Started exactly the way the templates do -- `java -jar`, so the manifest
# Class-Path resolves lib/ relative to the jar. Argument parsing alone needs
# commons-cli out of lib/, so this fails loudly when the dependencies are absent.
if java --add-modules=jdk.incubator.vector -jar "$jar" -s 'print("systemds-preflight-ok");' \
     2>/dev/null | grep -q systemds-preflight-ok; then
  note ok "SystemDS starts via java -jar"
else
  fail "SystemDS did not start from $jar (missing lib/, or a JVM/module problem)"
fi
command -v /opt/bench-venv/bin/spark-submit >/dev/null &&
  note ok "spark-submit present" ||
  note WARN "no spark-submit; the enabled systemds-spark template will abort the preflight
       (rebuild with INSTALL_SPARK=true, or disable that template)"

echo
echo "== python modules"
for module in numpy scipy sklearn dask.array distributed zarr sparse matplotlib yaml systemds; do
  if "$python" -c "import $module" >/dev/null 2>&1; then note ok "$module"
  elif [[ "$module" == systemds ]]; then
    note WARN "systemds (installed on first run from /opt/systemds/python)"
  else fail "$module"; fi
done

echo
echo "== data root"
if [[ -d "$root" && -w "$root" ]]; then
  free_gb="$(df -BG --output=avail "$root" | tail -1 | tr -dc '0-9')"
  if [[ "${free_gb:-0}" -lt 350 ]]; then
    note WARN "$root has ${free_gb}G free; the enabled sweep needs roughly 350G"
  else
    note ok "$root (${free_gb}G free)"
  fi
else
  fail "$root is not a writable directory"
fi

echo
echo "== plan"
container_plan="$plan_dir/benchmark-plan.container.yaml"
if "$python" "$plan_dir/docker/make_container_plan.py" \
     "$plan_dir/benchmark-plan.yaml" -o "$container_plan" >/dev/null 2>&1 &&
   "$python" "$plan_dir/benchmark_plan.py" "$container_plan" --validate; then
  :
else
  fail "plan validation"
fi

echo
[[ "$status" == 0 ]] && echo "Preflight passed; ./bench.sh run-detached is safe to start." ||
  echo "Preflight found blocking problems; see FAIL lines above." >&2
exit "$status"
