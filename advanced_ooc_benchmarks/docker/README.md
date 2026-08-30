# Containerized cgroup benchmark runner

Spin up one container and run the full suite:

```bash
cd advanced_ooc_benchmarks/docker
cp .env.example .env        # adjust paths, then
./bench.sh build
./bench.sh run              # == run_cgroup_baselines.sh, inside the container
```

`./bench.sh run` starts the container if needed, waits for the per-user systemd
manager, rewrites the plan for the container layout, and executes
`run_cgroup_baselines.sh`. A different plan is one argument away:

```bash
./bench.sh run /workspace/advanced_ooc_benchmarks/benchmark-plan2.yaml
```

Other commands: `./bench.sh up`, `./bench.sh shell`, `./bench.sh logs`,
`./bench.sh down`.

## Running on a remote host

`../sync_remote.sh [HOST] [REMOTE_DIR]` (defaults `so014` and `~/bench`) pushes the plan directory,
the jar from `tools.systemds_jar`, and that jar's `lib/` -- keeping the two siblings, since the
manifest `Class-Path` resolves `lib/...` relative to the jar -- and writes `docker/.env` on the
remote against its own uid and home. Nothing else needs to travel: the image builds its own Python
and spark-submit, and `make_container_plan.py` rewrites the plan's host paths at launch.

```bash
./sync_remote.sh so014 bench --data-dir /scratch/$USER/bench-data
```

The SystemDS Python bindings (`src/main/python` beside the jar) travel too, and unlike the jar
they are sent even with `--no-jar`: dataset preparation imports `systemds`, so the container
aborts before the first dataset without them.

`--dry-run` shows the transfer, `--no-jar` skips the ~350 MB build after the first sync, and
`--force-env` replaces an existing remote `.env`.

### First run on a new host

In order, because each step's failure mode is cheaper to see than the next one's:

```bash
./sync_remote.sh <host> bench --force-env --data-dir <data root on the fast mount>
ssh <host>
mkdir -p <data root>                      # if it is not under $HOME
cd ~/bench/advanced_ooc_benchmarks
./io_probe.sh <data root>                 # what does this storage actually deliver?
cd docker
docker compose version || echo "install the compose plugin into ~/.docker/cli-plugins"
./bench.sh build
./bench.sh preflight                      # cgroups, io.stat, tools, disk, plan
./bench.sh run-detached
```

`io_probe.sh` is not optional bookkeeping: on identical-looking nodes the data root has ranged
from 0.75 GB/s (an M.2 boot drive) to a 15-SSD RAID-0, and which engine wins depends on it.
Record the number beside the results.

## Preflight

```bash
./bench.sh preflight
```

Checks, from inside the container and without changing anything: the cgroup-v2 hierarchy and which
controllers are delegated to `user@<uid>.service`; that `systemd-run --user --scope` can actually
create a scope with `MemoryMax` and `IOAccounting`; whether `io.stat` is populated inside such a
scope; whether `/proc/sys/vm/drop_caches` is writable; the jar, its `lib/`, `spark-submit`, and that
SystemDS starts; every Python module the plan's implementations declare; free space against the
~350 GB the enabled sweep needs; and finally that the rewritten container plan validates.

`FAIL` lines block a run. `WARN` lines degrade the measurement without stopping it -- a missing
`io` controller costs byte-accurate I/O (the runner falls back to GNU time's rusage block counters),
and a read-only `drop_caches` means `drop_caches.py` uses per-file `POSIX_FADV_DONTNEED` instead.

The `io` controller is the one worth escalating, since `io_read_bytes` is what turns a runtime
difference into a count of passes over the input. It needs a host-side change, from an admin:

```ini
# /etc/systemd/system/user@.service.d/delegate.conf
[Service]
Delegate=cpu cpuset io memory pids
IOAccounting=yes
```

with `DefaultIOAccounting=yes` in `/etc/systemd/system.conf` so the controller is enabled in
`cgroup.subtree_control` from the root down. Accounting only -- no limits, no throttling. After
`systemctl daemon-reload` the delegation reaches a session only when its user manager restarts,
so log out of *every* session on the host and back in (or have the admin run
`loginctl terminate-user $USER`); `daemon-reload` alone changes nothing for an already-running
manager. Containers are not under `user@.service` at all, so also recreate the container
(`./bench.sh down && ./bench.sh up`) and, if the root hierarchy gained the controller,
restart the docker daemon. Re-run `./bench.sh preflight` to confirm rather than assuming.

Note that `drop_caches` cannot be fixed by privilege: `drop_caches.py` gates on write access from
the unprivileged runner, so making the file root-writable changes nothing. `io.pressure` is
unaffected by any of this -- PSI files exist in every v2 cgroup regardless of controller
delegation -- so the "fraction of wall time stalled on I/O" metric survives even the worst case.

## Long runs and where the results land

The full sweep takes hours, so start it detached rather than holding an ssh session open:

```bash
./bench.sh run-detached     # returns immediately
./bench.sh tail             # follow the newest run log
./bench.sh status           # is a sweep still in progress?
```

`run-detached` uses `docker exec -d`, so the run belongs to the docker daemon and survives the ssh
session that started it -- no `nohup`, nothing to keep open. Its console log goes to
`$BENCH_DATA_DIR/bench-results/run-<timestamp>.log`, which is bind-mounted and therefore readable
from the host while the run is still going. `./bench.sh run` also works non-interactively now (it
asks for a TTY only when it has one), so `nohup ./bench.sh run &` is a valid alternative.

Results are written under the plan's `results`, which is `$BENCH_DATA_DIR/bench-results`:

```
bench-results/<invocation-id>/
  benchmark-plan.yaml        the plan as executed, container paths and all
  expanded-plan.yaml         every case after dataset/profile/parameter expansion
  invocation-metadata.json   host, git revision, tool versions
  output-validation.json     cross-arm agreement per workload (written at the end)
  <run-case-id>/
    results.csv              the measurements: wall time, peak memory, cpu, io bytes/ops, oom events
    logs/                    per-execution .log, .metrics, .telemetry.csv, drop-caches.log
    outputs/                 models and vectors, pruned once the arms are found to agree
    resolved-run.json, resolved-context.json, output-retention.json
```

`results.csv` is flushed after every execution, so a run in progress can be inspected and a run
that dies partway still leaves usable rows. The invocation id is the start timestamp.

To pull an invocation back for plotting:

```bash
../download_results_remote.sh --host <host> --data-dir <remote data root>
python3 ../visualize_invocation.py ../results-remote/<invocation-id>
```

`download_results_remote.sh` fetches the newest invocation and its driver log by default
(`--list` to see what is there, `--invocation ID` to pick one, `--all` for every one,
`--no-telemetry` to skip the 1 Hz traces, which are the bulk of the tree). It never deletes, so
re-running it on a sweep still in progress just tops up the local copy. The visualizer writes
`runtime`, `read`, `write`, `cpu`, and `io` figures to `<workload>/results/<invocation-id>/`.

Keep the telemetry unless the transfer is painful: `<case>.telemetry.csv` samples `io.stat`,
`memory.stat` and the PSI totals once a second, so the cumulative `io_read_bytes` column
differentiates into a bandwidth-over-time trace -- which is how you tell a run that was I/O-bound
throughout from one that stalled in a single phase.

## Why the container looks the way it does

The suite is not a plain batch job: every implementation runs inside its own
cgroup-v2 scope created with `systemd-run --user --scope`, and the measurements
are read straight out of that scope's `memory.*`, `cpu.stat`, `io.stat`, and
`*.pressure` files. That dictates three things.

- **systemd is PID 1** and the benchmark user has lingering enabled, so
  `systemd-run --user` has a manager to talk to. `benchmark_plan.py` aborts up
  front if `systemctl --user status` fails.
- **The container is privileged** with a writable cgroup hierarchy. Privileged
  also makes `/proc/sys/vm/drop_caches` writable, which is what lets
  `drop_caches.py` take the reliable path instead of per-file
  `POSIX_FADV_DONTNEED`. Note that this drops the **host's** page cache too.
- **`user@.service` gets `Delegate=cpu cpuset io memory pids`.** Without the
  `io` controller the scope has no `io.stat` and the runner warns that
  byte-accurate I/O will not be recorded.

## What is mounted, and what is not

| Host | Container | Purpose |
| --- | --- | --- |
| `REPO_DIR` (default `../..`) | `/workspace` | plan directory, DML/NumPy/Dask sources, runner scripts |
| `BENCH_DATA_DIR` (default `./data`) | `/bench/data` | the plan's `root`: `bench-data/`, `bench-results/`, `tmp/`, `scratch_space/` |
| `SYSTEMDS_JAR` | `/opt/systemds/SystemDS.jar` | the SystemDS build under test |
| `SYSTEMDS_LIB_DIR` | `/opt/systemds/lib` | its dependencies; the jar's manifest `Class-Path` resolves `lib/...` relative to the jar, so it cannot start without them |

The image installs a JDK and the baseline Python stack
(`requirements-baselines.txt` in `/opt/bench-venv`) but deliberately
**does not build SystemDS**: the jar is the thing being measured and usually
comes from a working checkout, so it is mounted read-only. Build it on the host
with `mvn -q -DskipTests package` and point `SYSTEMDS_JAR` at
`target/SystemDS.jar`.

Put `BENCH_DATA_DIR` on a real block device with room for the dense sweep
(tens of GB once the blocksize-qualified SystemDS representations exist).
Datasets missing from it are generated on first run by the plan's `prepare`
commands, which takes a while but needs no manual step. An overlay-only path
also gives no meaningful `io.stat` attribution.

## Plan rewriting

Plans carry absolute host paths in exactly five places: `root`, the three
`tools` entries, and `environment.SPARK_HOME`. Everything else derives from
`${plan.root}` and `${tools.*}`. `make_container_plan.py` rewrites those lines
into `benchmark-plan.container.yaml` next to the source plan (git-ignored), and
`run-in-container.sh` points `BENCHMARK_PLAN` at the result. Comments and every
other byte of the plan are preserved, so the copy stored in the invocation
directory stays diffable against the source plan.

The generated plan lives in the plan directory because `benchmark_plan.py`
resolves `${plan.dir}`, the entrypoints, and the preparation scripts relative to
the plan file.

## Memory

The shipped profiles cap runs at 16/8/4 GB with a 12g/6g/3g JVM heap. The
container itself is intentionally left unlimited — an outer `mem_limit` would
sit above the scopes and distort exactly what is being measured — so the host
needs enough free RAM for the largest profile. On Docker Desktop, raise the VM's
memory allocation; a Linux host needs no extra step. To run on a smaller
machine, drop the larger entries from each run's `resource_profiles` list.

## Spark

The local Spark arm is disabled in the shipped plan. To use it, build with
`INSTALL_SPARK=true` (`.env`), which installs PySpark into the same venv; the
plan generator then also rewrites `SPARK_HOME` to that package. Re-enable
`systemds-spark` in the plan's `templates`.

## Troubleshooting

- *"A working user systemd instance and systemd-run are required"* — the user
  manager did not start. `./bench.sh logs`, then
  `docker exec -u root advanced-ooc-benchmarks systemctl start user@1000.service`.
- *"cgroup io.stat is unavailable"* — the `io` controller is not reaching the
  scope. Check
  `cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/cgroup.controllers`
  inside the container; it must list `io`. Some kernels also need `io` enabled in
  the host root cgroup's `cgroup.subtree_control`.
- *Results owned by root on the host* — rebuild with `BENCH_UID`/`BENCH_GID` set
  to your own ids.
- *cgroup v1 host* — the suite requires the unified hierarchy. Boot the host with
  `systemd.unified_cgroup_hierarchy=1`.
