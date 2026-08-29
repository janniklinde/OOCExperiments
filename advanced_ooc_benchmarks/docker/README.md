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
