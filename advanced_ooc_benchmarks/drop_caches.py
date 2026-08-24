#!/usr/bin/env python3
"""Evict the page cache for the benchmark inputs so each cgroup scope starts cold.

cgroup v2 charges a page-cache page to the cgroup that first faults it in, and keeps it
charged there. A scope that maps a file whose pages are already resident therefore gets
them for free: they do not count against its MemoryMax. That is how the numpy-memmap
baseline ran with the whole 3.7 GB matrix resident while its cgroup reported a 35 MB peak.

Global /proc/sys/vm/drop_caches is the reliable way to do this but needs root. Without it,
POSIX_FADV_DONTNEED drops the clean, unmapped pages of the named files, which is enough for
read-only inputs between runs. Pass directories to walk them recursively.

Usage: drop_caches.py PATH [PATH ...]
"""
import ctypes
import os
import sys

POSIX_FADV_DONTNEED = 4

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]


def cached_pages(path):
    """Resident pages of path, via mincore, so the caller can verify the drop worked."""
    import mmap
    size = os.path.getsize(path)
    if size == 0:
        return 0
    with open(path, "rb") as fh:
        #MAP_PRIVATE rather than ACCESS_READ: ctypes needs a writable buffer to take the address,
        #and copy-on-write shares the same pages until something writes to them
        mm = mmap.mmap(fh.fileno(), size, mmap.MAP_PRIVATE)
        try:
            pagesize = os.sysconf("SC_PAGE_SIZE")
            npages = (size + pagesize - 1) // pagesize
            vec = (ctypes.c_ubyte * npages)()
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return -1
            return sum(1 for b in vec if b & 1)
        finally:
            mm.close()


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)  # DONTNEED skips dirty pages, so write them back first
        except OSError:
            pass  # read-only fd on a filesystem that refuses fsync; inputs are clean anyway
        rc = _libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if rc != 0:
        raise OSError(rc, "posix_fadvise failed on " + path)


def walk(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in names:
                    yield os.path.join(root, name)
        elif os.path.isfile(path):
            yield path


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if os.access("/proc/sys/vm/drop_caches", os.W_OK):
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3\n")
        print("dropped all caches via /proc/sys/vm/drop_caches")
        return 0
    pagesize = os.sysconf("SC_PAGE_SIZE")
    residual = 0
    for path in walk(argv[1:]):
        try:
            drop(path)
            left = cached_pages(path)
        except OSError as ex:
            print("warn: %s: %s" % (path, ex), file=sys.stderr)
            continue
        if left > 0:
            residual += left
            print("warn: %s still has %d MiB cached (mapped elsewhere?)"
                  % (path, left * pagesize >> 20), file=sys.stderr)
    if residual:
        print("residual cache: %d MiB -- runs will not be fully cold"
              % (residual * pagesize >> 20), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
