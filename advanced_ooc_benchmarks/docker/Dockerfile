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
        bash gawk coreutils time procps util-linux \
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
    && touch /var/lib/systemd/linger/${BENCH_USER}

# Baseline Python stack. The requirements file carries pyyaml because
# benchmark_plan.py itself runs under the plan's configured `tools.python`.
COPY requirements-baselines.txt /tmp/requirements-baselines.txt
RUN python3 -m venv /opt/bench-venv \
    && /opt/bench-venv/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /opt/bench-venv/bin/pip install --no-cache-dir -r /tmp/requirements-baselines.txt \
    && if [ "${INSTALL_SPARK}" = "true" ]; then \
         /opt/bench-venv/bin/pip install --no-cache-dir "pyspark==${PYSPARK_VERSION}"; \
       fi \
    && rm /tmp/requirements-baselines.txt

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
