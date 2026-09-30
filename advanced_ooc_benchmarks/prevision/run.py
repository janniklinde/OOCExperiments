#!/usr/bin/env python3
"""Run the published PreVision engine under the suite's timed cgroup wrapper."""

import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SHM_FILES = tuple(Path("/dev/shm") / f"buffertile_{name}"
                  for name in ("data", "idata", "key", "bf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mode", choices=("gnmf", "gram"), required=True)
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--w", type=Path)
    parser.add_argument("--h", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--cache-bytes", type=int, required=True)
    parser.add_argument("--metadata-bytes", type=int, default=128 * 1024 * 1024,
                        help="size of each of PreVision's two metadata stores")
    parser.add_argument("--arena", choices=("posix", "memfd"), default="posix",
                        help="memfd keeps Linux shmem accounting without /dev/shm mount quota")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--elementwise-threads", type=int,
                        help="upstream elementwise pthread count (default: same as --threads)")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--engine-log", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.elementwise_threads is None:
        args.elementwise_threads = args.threads
    for field in ("adapter", "x", "w", "h", "work_dir", "output_prefix",
                  "engine_log", "lock"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if args.cache_bytes < 16 * 1024 * 1024 or args.metadata_bytes < 4 * 1024 * 1024 or min(args.threads, args.elementwise_threads) < 1:
        parser.error("cache must be at least 16 MiB, metadata at least 4 MiB, and threads positive")
    if args.mode == "gnmf" and (not args.w or not args.h):
        parser.error("GNMF requires --w and --h")
    if args.arena == "posix":
        required_shm = args.cache_bytes + 2 * args.metadata_bytes
        shm = os.statvfs("/dev/shm")
        if shm.f_blocks * shm.f_frsize < required_shm:
            parser.error(f"/dev/shm is too small: need at least {required_shm} bytes; "
                         "increase the container's SHM_SIZE or select --arena memfd")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.engine_log.parent.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    # TileStore's API takes a logical array name and appends `.tilestore` on disk.
    # The plan tracks the physical directory so readiness checks can inspect it.
    def logical_name(path: Path) -> str:
        if path.suffix != ".tilestore" or not path.is_dir():
            parser.error(f"expected a prepared TileStore directory: {path}")
        return str(path.with_suffix(""))

    command = [str(args.adapter), args.mode, logical_name(args.x)]
    if args.mode == "gnmf":
        command += [logical_name(args.w), logical_name(args.h), str(args.rank),
                    str(args.iterations), str(args.epsilon)]
    command.append(str(args.output_prefix))
    env = dict(os.environ)
    env.update({
        "BF_DATA_SIZE": str(args.cache_bytes),
        "BF_IDATA_SIZE": "0",
        "BF_KEYSTORE_SIZE": str(args.metadata_bytes),
        "BF_BFSTORE_SIZE": str(args.metadata_bytes),
        "BF_EVICTION_POLICY": "8",  # Published OPT replacement policy.
        "BF_LRUK_K": "0", "BF_LRUK_CRP": "0", "BF_PREEMPTIVE_EVICTION": "1",
        "__PREVISION_NUM_THREADS": str(args.elementwise_threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
    })
    if args.arena == "memfd":
        shim = args.adapter.parent / "memfd_shm.so"
        if not shim.is_file():
            parser.error(f"missing Linux memfd shim: {shim}; rebuild the adapter")
        env["LD_PRELOAD"] = str(shim) + (" " + env["LD_PRELOAD"]
                                            if env.get("LD_PRELOAD") else "")
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    # PreVision uses four fixed POSIX shared-memory names. Serialize this suite's
    # PreVision runs and clear only those names after a previously killed run.
    with args.lock.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.arena == "posix":
            for path in SHM_FILES:
                path.unlink(missing_ok=True)
        first = bytearray()
        tail: deque[bytes] = deque()
        tail_size = 0
        next_checkpoint = time.monotonic() + 60

        def checkpoint() -> None:
            # Keep periodic disk writes small; save the full tail only on exit.
            suffix = b"".join(tail)
            if process is not None and process.poll() is None:
                suffix = suffix[-65536:]
            omitted = max(0, total_bytes - len(first) - len(suffix))
            marker = (f"\n[PreVision log truncated: {omitted} intermediate bytes]\n".encode()
                      if omitted else b"")
            args.engine_log.write_bytes(bytes(first) + marker + suffix)

        total_bytes = 0
        process = None
        try:
            print(f"PreVision {args.mode} ({args.arena} arena, BLAS threads={args.threads}, "
                  f"elementwise threads={args.elementwise_threads}): engine log {args.engine_log}", flush=True)
            process = subprocess.Popen(command, cwd=args.work_dir, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.stdout is not None
            while chunk := process.stdout.read(65536):
                total_bytes += len(chunk)
                first_part = min(len(chunk), max(0, 256 * 1024 - len(first)))
                first.extend(chunk[:first_part])
                remainder = chunk[first_part:]
                if remainder:
                    tail.append(remainder)
                    tail_size += len(remainder)
                while tail_size > 4 * 1024 * 1024:
                    removed = tail.popleft()
                    tail_size -= len(removed)
                if time.monotonic() >= next_checkpoint:
                    checkpoint()
                    next_checkpoint = time.monotonic() + 60
            rc = process.wait()
            if rc:
                print(f"PreVision exited {rc}; see {args.engine_log}", file=sys.stderr)
            else:
                print(args.output_prefix.with_suffix(".json").read_text().strip())
            return rc
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if total_bytes:
                checkpoint()
            if args.arena == "posix":
                for path in SHM_FILES:
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
