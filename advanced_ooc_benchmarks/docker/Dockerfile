# syntax=docker/dockerfile:1
#
# Container image for the standalone advanced OOC benchmark suite.
#
# The suite measures every run inside its own cgroup-v2 scope created with
# `systemd-run --user --scope`, so the image runs a real systemd as PID 1 and a
# per-user systemd manager for the benchmark user. Everything else the runner
# shells out to - java, /usr/bin/time, timeout, awk, the baseline Python stack -
# is installed here so that `run_cgroup_baselines.sh` works with no host setup
# beyond the mounted dataset directory and SystemDS jar.
#
# Build context is the parent directory (advanced_ooc_benchmarks), so the
# baseline requirements file can be copied in.
FROM ubuntu:24.04

ARG BENCH_USER=bench
ARG BENCH_UID=1000
ARG BENCH_GID=1000
ARG JAVA_VERSION=21
# Spark is disabled in the shipped plan (`systemds-spark: enabled: false`).
# Build with `--build-arg INSTALL_SPARK=true` to enable the local[N] Spark arm.
ARG INSTALL_SPARK=false
ARG PYSPARK_VERSION=3.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    container=docker

RUN apt-get update && apt-get install -y --no-install-recommends \
        systemd systemd-sysv dbus dbus-user-session \
        openjdk-${JAVA_VERSION}-jdk-headless \
        python3 python3-venv \
        bash gawk coreutils time procps util-linux sudo \
        ca-certificates curl git tzdata less \
    && rm -rf /var/lib/apt/lists/*

# Units that cannot work (or are pointless) inside a container. Leaving them
# enabled only slows boot down and fills the journal with failures.
RUN systemctl mask \
        systemd-udevd.service systemd-udev-trigger.service \
        systemd-modules-load.service systemd-networkd.service \
        systemd-resolved.service systemd-timesyncd.service \
        systemd-firstboot.service \
        sys-kernel-debug.mount sys-kernel-config.mount sys-kernel-tracing.mount \
        dev-hugepages.mount console-getty.service getty.target

# The user systemd manager only gets the controllers that PID 1 delegates to it.
# Without `io` the scope has no io.stat and benchmark_plan.py degrades to
# "byte-accurate read/write I/O metrics will not be recorded".
RUN mkdir -p /etc/systemd/system/user@.service.d \
    && printf '[Service]\nDelegate=cpu cpuset io memory pids\n' \
        > /etc/systemd/system/user@.service.d/delegate.conf

# cgroup v2 refuses to enable a controller in a cgroup that holds processes, and a
# container's namespaced cgroup root is not the real root, so it gets no exemption.
# `docker exec` places every process it starts in exactly that root, so by the time
# anyone execs in, the enable is already impossible -- which leaves subtree_control
# empty all the way down and makes `systemd-run --user --scope -p MemoryMax=` a no-op
# that is accepted and never enforced. Doing it at boot, before anything execs in, is
# the only reliable moment.
RUN printf '%s\n' \
    '#!/bin/sh' \
    '# Enable the cgroup v2 controllers the benchmark scopes need. Idempotent.' \
    'root=/sys/fs/cgroup' \
    '[ -w "$root/cgroup.subtree_control" ] || exit 0' \
    'mkdir -p "$root/exec-holding"' \
    'for p in $(cat "$root/cgroup.procs" 2>/dev/null); do' \
    '  [ "$p" = 1 ] || echo "$p" > "$root/exec-holding/cgroup.procs" 2>/dev/null || true' \
    'done' \
    'for dir in "$root" "$root/user.slice"; do' \
    '  [ -d "$dir" ] || continue' \
    '  for c in memory io pids cpu; do' \
    '    echo "+$c" > "$dir/cgroup.subtree_control" 2>/dev/null || true' \
    '  done' \
    'done' \
    'exit 0' \
        > /usr/local/bin/bench-enable-controllers \
    && chmod 0755 /usr/local/bin/bench-enable-controllers \
    && printf '%s\n' \
    '[Unit]' \
    'Description=Enable cgroup v2 controllers in the container cgroup root' \
    'ConditionVirtualization=container' \
    'Before=user.slice' \
    '' \
    '[Service]' \
    'Type=oneshot' \
    'RemainAfterExit=yes' \
    'ExecStart=/usr/local/bin/bench-enable-controllers' \
    '' \
    '[Install]' \
    'WantedBy=multi-user.target' \
        > /etc/systemd/system/bench-cgroup-controllers.service \
    && systemctl enable bench-cgroup-controllers.service

# Delegation alone is not enough: systemd enables a controller in a slice's
# cgroup.subtree_control only when something below it asks for that controller.
# Without these defaults user.slice and user-<uid>.slice can leave memory and io
# disabled, and then `systemd-run --user --scope -p MemoryMax=` is accepted and
# silently unenforced -- a run that looks fine and measures nothing.
RUN mkdir -p /etc/systemd/system.conf.d \
    && printf '[Manager]\nDefaultCPUAccounting=yes\nDefaultIOAccounting=yes\nDefaultMemoryAccounting=yes\nDefaultTasksAccounting=yes\n' \
        > /etc/systemd/system.conf.d/accounting.conf

# /proc/sys/vm/drop_caches is root-owned 0644, so the benchmark user cannot write
# it even here -- `privileged: true` grants the container capabilities, not this
# unprivileged uid. One NOPASSWD helper restores the global cache drop; inside a
# privileged, disposable container it concedes nothing that is not already root.
RUN printf '#!/bin/sh\nsync\nexec tee /proc/sys/vm/drop_caches\n' \
        > /usr/local/bin/bench-drop-caches \
    && chmod 0755 /usr/local/bin/bench-drop-caches

# Ubuntu 24.04 ships a stock uid 1000 user; drop it so the benchmark user can
# take the host uid and keep bind-mounted results writable from the host.
RUN if getent passwd ${BENCH_UID} >/dev/null; then \
        userdel -r "$(getent passwd ${BENCH_UID} | cut -d: -f1)" 2>/dev/null || true; \
    fi \
    && if getent group ${BENCH_GID} >/dev/null; then \
        groupmod -n ${BENCH_USER} "$(getent group ${BENCH_GID} | cut -d: -f1)"; \
    else groupadd -g ${BENCH_GID} ${BENCH_USER}; fi \
    && useradd -m -u ${BENCH_UID} -g ${BENCH_GID} -s /bin/bash ${BENCH_USER} \
    && mkdir -p /var/lib/systemd/linger \
    && touch /var/lib/systemd/linger/${BENCH_USER} \
    && printf '%s ALL=(root) NOPASSWD: /usr/local/bin/bench-drop-caches\n' ${BENCH_USER} \
        > /etc/sudoers.d/bench-drop-caches \
    && chmod 0440 /etc/sudoers.d/bench-drop-caches

# Baseline Python stack. The requirements file carries pyyaml because
# benchmark_plan.py itself runs under the plan's configured `tools.python`, and
# py4j/pandas/requests because the SystemDS Python bindings need them. The
# bindings themselves cannot be installed here -- they live in the SystemDS
# checkout, outside this build context -- so they are mounted and installed on
# first use by run-in-container.sh, which is why the venv ends up user-owned.
COPY requirements-baselines.txt /tmp/requirements-baselines.txt
RUN python3 -m venv /opt/bench-venv \
    && /opt/bench-venv/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /opt/bench-venv/bin/pip install --no-cache-dir -r /tmp/requirements-baselines.txt \
    && if [ "${INSTALL_SPARK}" = "true" ]; then \
         /opt/bench-venv/bin/pip install --no-cache-dir "pyspark==${PYSPARK_VERSION}"; \
       fi \
    && rm /tmp/requirements-baselines.txt \
    && chown -R ${BENCH_UID}:${BENCH_GID} /opt/bench-venv

# Mount points: the repository, the benchmark data/results root, and the jar.
RUN mkdir -p /workspace /bench/data /opt/systemds \
    && chown ${BENCH_UID}:${BENCH_GID} /bench/data

ENV JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64 \
    BENCH_USER=${BENCH_USER} \
    BENCH_UID=${BENCH_UID} \
    BENCH_CONTAINER_ROOT=/bench/data \
    BENCH_CONTAINER_PYTHON=/opt/bench-venv/bin/python \
    BENCH_CONTAINER_JAR=/opt/systemds/SystemDS.jar \
    BENCH_PLAN_DIR=/workspace/advanced_ooc_benchmarks

WORKDIR /workspace/advanced_ooc_benchmarks

# systemd's shutdown signal.
STOPSIGNAL SIGRTMIN+3
ENTRYPOINT ["/sbin/init"]
