/* Optional Linux container shim for PreVision's fixed BufferTile shm names.
 * memfd uses the same kernel shmem backing without the container's tiny
 * /dev/shm mount quota. Only BufferTile names are intercepted. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static struct {
    const char *name;
    int fd;
} entries[] = {
    {"buffertile_data", -1},
    {"buffertile_idata", -1},
    {"buffertile_key", -1},
    {"buffertile_bf", -1},
};
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static int entry_for(const char *name) {
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); ++i)
        if (strcmp(name, entries[i].name) == 0)
            return (int)i;
    return -1;
}

int shm_open(const char *name, int flags, mode_t mode) {
    int entry = entry_for(name);
    if (entry < 0) {
        int (*original)(const char *, int, mode_t) = dlsym(RTLD_NEXT, "shm_open");
        if (original) return original(name, flags, mode);
        errno = ENOSYS;
        return -1;
    }
    pthread_mutex_lock(&mutex);
    int existing = entries[entry].fd;
    if (existing >= 0 && (flags & O_CREAT) && (flags & O_EXCL)) {
        pthread_mutex_unlock(&mutex);
        errno = EEXIST;
        return -1;
    }
    if (existing < 0 && !(flags & O_CREAT)) {
        pthread_mutex_unlock(&mutex);
        errno = ENOENT;
        return -1;
    }
    if (existing < 0) {
        existing = memfd_create(entries[entry].name, MFD_CLOEXEC);
        if (existing < 0) {
            pthread_mutex_unlock(&mutex);
            return -1;
        }
        entries[entry].fd = existing;
    }
    int fd = fcntl(existing, F_DUPFD_CLOEXEC, 0);
    pthread_mutex_unlock(&mutex);
    return fd;
}

int shm_unlink(const char *name) {
    int entry = entry_for(name);
    if (entry < 0) {
        int (*original)(const char *) = dlsym(RTLD_NEXT, "shm_unlink");
        if (original) return original(name);
        errno = ENOSYS;
        return -1;
    }
    pthread_mutex_lock(&mutex);
    int fd = entries[entry].fd;
    entries[entry].fd = -1;
    pthread_mutex_unlock(&mutex);
    if (fd < 0) {
        errno = ENOENT;
        return -1;
    }
    return close(fd);
}
