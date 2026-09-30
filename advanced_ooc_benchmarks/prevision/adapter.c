/* Adapter for PreVision's published TileStore and lazy array API.
 * No PreVision source is copied into this benchmark suite. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tilestore.h"
#include "bf.h"
#include "exec_interface.h"
#include "node_interface.h"
#include "lam_interface.h"
#include "workloads.h"

static void die(const char *message) {
    fprintf(stderr, "prevision adapter: %s\n", message);
    exit(1);
}

static uint64_t positive(const char *value) {
    char *end = NULL;
    errno = 0;
    uint64_t parsed = strtoull(value, &end, 10);
    if (errno || !value[0] || *end || !parsed)
        die("expected a positive integer");
    return parsed;
}

static double number(const char *value) {
    char *end = NULL;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno || !value[0] || *end || !isfinite(parsed))
        die("expected a finite number");
    return parsed;
}

/* Convert canonical row-major FP64 into FP64 TileStore, one tile at a time. */
static void import_fp64(int argc, char **argv) {
    if (argc != 10) die("import RAW ARRAY ROWS COLS TILE_ROWS TILE_COLS MODE SEED");
    const char *raw = argv[2], *array = argv[3], *mode = argv[8];
    uint64_t rows = positive(argv[4]), cols = positive(argv[5]);
    uint64_t tr = positive(argv[6]), tc = positive(argv[7]);
    uint64_t seed = positive(argv[9]);
    if (strcmp(mode, "copy") && strcmp(mode, "gnmf-w") && strcmp(mode, "gnmf-h"))
        die("MODE must be copy, gnmf-w, or gnmf-h");
    if (rows % tr || cols % tc)
        die("initial adapter requires dimensions divisible by tile extents");
    if (tr > SIZE_MAX / tc / sizeof(double)) die("tile too large");
    size_t tile_bytes = (size_t)tr * tc * sizeof(double);
    double *tile = NULL;
    if (posix_memalign((void **)&tile, 512, ceil_to_512bytes(tile_bytes)))
        die("tile allocation failed");
    FILE *input = NULL;
    if (!strcmp(mode, "copy")) {
        struct stat st;
        if (stat(raw, &st) || (uint64_t)st.st_size != rows * cols * sizeof(double))
            die("raw file missing or does not match declared FP64 shape");
        input = fopen(raw, "rb");
        if (!input) die("cannot open raw FP64 file");
    }
    uint64_t shape[2] = {rows, cols}, extents[2] = {tr, tc};
    if (tilestore_create_array(array, shape, extents, 2, TILESTORE_FLOAT64,
                               TILESTORE_DENSE) != TILESTORE_OK)
        die("cannot create TileStore array (does it already exist?)");
    for (uint64_t ri = 0; ri < rows / tr; ri++) {
        for (uint64_t ci = 0; ci < cols / tc; ci++) {
            if (input && tc == cols) {
                if (fread(tile, 1, tile_bytes, input) != tile_bytes)
                    die("cannot read contiguous source tile");
            } else {
                for (uint64_t r = 0; r < tr; r++) {
                    double *line = tile + r * tc;
                    if (input) {
                        if (fseeko(input, (off_t)(((ri * tr + r) * cols + ci * tc) * 8), SEEK_SET)
                            || fread(line, sizeof(double), tc, input) != tc)
                            die("cannot read source row");
                    } else {
                        for (uint64_t c = 0; c < tc; c++) {
                            uint64_t row = ri * tr + r + 1, col = ci * tc + c + 1;
                            line[c] = !strcmp(mode, "gnmf-w")
                                ? 0.01 + (double)((row * col + seed) % 97) / 97.0
                                : 0.01 + (double)((row * col + 3 * seed) % 89) / 89.0;
                        }
                    }
                }
            }
            uint64_t coords[2] = {ri, ci};
            if (tilestore_write_dense(array, coords, 2, tile, tile_bytes,
                                      TILESTORE_DIRECT) != TILESTORE_OK)
                die("cannot write dense TileStore tile");
        }
        if ((ri + 1) % 1000 == 0)
            fprintf(stderr, "imported %" PRIu64 "/%" PRIu64 " tile rows\n",
                    ri + 1, rows / tr);
    }
    if (input) fclose(input);
    free(tile);
}

/* Materialize every output element within the timed scope. */
static double export_array(Array *a, const char *path) {
    if (a->desc.dim_len != 2 || a->desc.array_type != TILESTORE_DENSE)
        die("expected dense two-dimensional output");
    uint64_t rows = a->desc.dim_domains[0][1] - a->desc.dim_domains[0][0] + 1;
    uint64_t cols = a->desc.dim_domains[1][1] - a->desc.dim_domains[1][0] + 1;
    uint64_t tr = a->desc.tile_extents[0], tc = a->desc.tile_extents[1];
    if (rows % tr || cols % tc) die("output has partial edge tiles");
    FILE *out = fopen(path, "wb+");
    if (!out) die("cannot open output file");
    double *values = NULL;
    if (posix_memalign((void **)&values, 512,
                       ceil_to_512bytes(tr * tc * sizeof(double))))
        die("output tile allocation failed");
    double checksum = 0;
    for (uint64_t ri = 0; ri < rows / tr; ri++) {
        for (uint64_t ci = 0; ci < cols / tc; ci++) {
            uint64_t coords[2] = {ri, ci};
            tilestore_tile_metadata_t metadata;
            if (tilestore_read_metadata_of_tile(a->desc.object_name, coords, 2,
                                                &metadata) != TILESTORE_OK
                || metadata.data_nbytes != tr * tc * sizeof(double)
                || tilestore_read_dense(a->desc.object_name, metadata, values,
                                        metadata.data_nbytes, TILESTORE_NORMAL) != TILESTORE_OK)
                die("PreVision did not materialize a complete output tile");
            for (uint64_t r = 0; r < tr; r++) {
                if (fseeko(out, (off_t)(((ri * tr + r) * cols + ci * tc) * 8), SEEK_SET)
                    || fwrite(values + r * tc, sizeof(double), tc, out) != tc)
                    die("cannot write output row");
                for (uint64_t c = 0; c < tc; c++) checksum += values[r * tc + c];
            }
        }
    }
    free(values);
    if (fclose(out)) die("cannot close output file");
    return checksum;
}

static void report(const char *path, const char *kind, double a, double b) {
    FILE *out = fopen(path, "w");
    if (!out) die("cannot write report");
    if (!strcmp(kind, "gnmf"))
        fprintf(out, "{\"implementation\":\"prevision-gnmf\","
                     "\"w_checksum\":%.17g,\"h_checksum\":%.17g}\n", a, b);
    else
        fprintf(out, "{\"implementation\":\"prevision-gram\","
                     "\"gram_checksum\":%.17g}\n", a);
    if (fclose(out)) die("cannot close report");
}

static void start_buffer(void) {
    if (BF_Init() != BFE_OK || BF_Attach() != BFE_OK)
        die("PreVision buffer initialization failed; check BF_* sizes and /dev/shm");
}

static void gnmf(int argc, char **argv) {
    if (argc != 9) die("gnmf X W H RANK ITERATIONS EPSILON OUT_PREFIX");
    uint64_t rank = positive(argv[5]), iterations = positive(argv[6]);
    double epsilon = number(argv[7]);
    if (epsilon <= 0) die("epsilon must be positive");
    start_buffer();
    Array *X = open_array(argv[2]), *W = open_array(argv[3]), *H = open_array(argv[4]);
    if (!X || !W || !H) die("cannot open PreVision input array");
    if (H->desc.dim_domains[0][1] + 1 != rank) die("H rank mismatch");
    prevision_gnmf(X, &W, &H, iterations, &epsilon);
    W->persist = true;
    H->persist = true;
    execute(W);
    execute(H);
    char path[4096];
    snprintf(path, sizeof(path), "%s-W.f64", argv[8]);
    double w_sum = export_array(W, path);
    snprintf(path, sizeof(path), "%s-H.f64", argv[8]);
    double h_sum = export_array(H, path);
    snprintf(path, sizeof(path), "%s.json", argv[8]);
    report(path, "gnmf", w_sum, h_sum);
    printf("w_checksum=%.17g h_checksum=%.17g\n", w_sum, h_sum);
    BF_Detach(); BF_Free();
}

static void gram(int argc, char **argv) {
    if (argc != 4) die("gram X OUT_PREFIX");
    start_buffer();
    Array *X = open_array(argv[2]);
    if (!X) die("cannot open PreVision input array");
    Array *G = prevision_gram(X);
    G->persist = true;
    execute(G);
    char path[4096];
    snprintf(path, sizeof(path), "%s-G.f64", argv[3]);
    double g_sum = export_array(G, path);
    snprintf(path, sizeof(path), "%s.json", argv[3]);
    report(path, "gram", g_sum, 0);
    printf("gram_checksum=%.17g\n", g_sum);
    BF_Detach(); BF_Free();
}

int main(int argc, char **argv) {
    if (argc < 2) die("command required: import, gnmf, or gram");
    if (!strcmp(argv[1], "import")) import_fp64(argc, argv);
    else if (!strcmp(argv[1], "gnmf")) gnmf(argc, argv);
    else if (!strcmp(argv[1], "gram")) gram(argc, argv);
    else die("unknown command");
    return 0;
}
