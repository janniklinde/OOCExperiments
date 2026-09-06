"""Keep the OS page cache clear of a dataset directory while a benchmark runs.

Step 2 of OOC-READ-PATH-PLAN.md: the OOC cache already holds hot blocks decoded in the Java
heap, so the kernel's page cache holds a second, encoded copy of the same data inside the same
cgroup budget. Under a memory cap that second cache does negative work -- it wins few hits, it
steals memory from the heap, and it generates readahead that is evicted before use. On the
mem16 profile this showed up as 641.4 GB of device reads for 285.1 GB of engine demand (2.25x)
plus 152.8 GB of workingset refaults.

This drops the clean page cache behind the dataset with POSIX_FADV_DONTNEED on a loop, which is
the same syscall an in-engine fix would issue, just from outside the JVM. It validates the fix
before committing to the JNI/FFM work needed to call fadvise from SystemDS itself.

usage:  python3 pagecache_evictor.py <dataset_dir> [interval_seconds]
        run it in the background for the duration of a benchmark run, then compare
        io_read_bytes and wall time against a run without it.
"""
import os
import signal
import sys
import time

running = True


def stop(_signum, _frame):
    global running
    running = False


def files_under(root):
    out = []
    for base, _, names in os.walk(root):
        for n in names:
            out.append(os.path.join(base, n))
    return out


def drop(paths):
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return n


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    rounds = 0
    start = time.time()
    while running:
        paths = files_under(root)
        if not paths:
            print("no files under %s" % root, file=sys.stderr)
            return 1
        drop(paths)
        rounds += 1
        time.sleep(interval)
    print("evictor: %d rounds over %.1f s on %s" % (rounds, time.time() - start, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
