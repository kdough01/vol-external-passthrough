/* ============================================================================
 * bench_pressio_timing.c   --  APPROACH 2 of 3: libpressio DIRECT (no HDF5)
 * ----------------------------------------------------------------------------
 * The lower-bound comparator. It runs the exact same codec configuration as the
 * VOL and the filter, but with zero HDF5/VOL machinery: compress in memory, then
 * write the compressed payload straight to a file with fwrite. Subtracting this
 * TOTAL from the VOL/filter TOTAL is what isolates the abstraction's overhead.
 *
 * Phases here:
 *   COMPRESS  = pressio_compressor_compress / _decompress   (GPU events or steady)
 *   IO        = fwrite / fread of the compressed payload
 *   OVERHEAD  = ~0 by construction (there is no wrapper) -> sanity check
 *   TOTAL     = COMPRESS + IO
 *
 * Build (CPU only):
 *   cc  -O2 -std=gnu11 bench_pressio_timing.c -lpressio -o bench_pressio_timing
 * Build (with GPU-event timing for cuszp):
 *   nvcc -O2 -x cu -DBENCH_TIMING_WITH_CUDA bench_pressio_timing.c \
 *        -lpressio -lcudart -o bench_pressio_timing
 *
 * NOTE on the JSON call: this uses pressio_options_new_json(), matching the
 * connector's option-replay path. If your libpressio build exposes it under a
 * different name, swap it in one place (make_options below).
 * ==========================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <libpressio.h>
#ifdef BENCH_TIMING_WITH_CUDA
#include <cuda_runtime.h>
#endif

#define BENCH_DATASETS_ENABLE_PRESSIO   /* dtype + reversed-dims helpers */
#include "bench_datasets.h"
#include "bench_compressors.h"
#include "bench_timing.h"

/* Load one raw field into a host buffer. */
static size_t load_field(const bench_dataset_t *d, void **out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = malloc(nbytes);
    if (!buf) return 0;
    FILE *f = fopen(d->path, "rb");
    if (!f) { free(buf); return 0; }
    size_t got = fread(buf, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) { free(buf); return 0; }
    *out = buf;
    return nbytes;
}

/* Configure a compressor for (compressor, dataset): options + (GPU) stream. */
static struct pressio_compressor *
make_compressor(struct pressio *library, const bench_compressor_t *c,
                const bench_dataset_t *d
#ifdef BENCH_TIMING_WITH_CUDA
                , cudaStream_t stream
#endif
                ) {
    struct pressio_compressor *comp = pressio_get_compressor(library, c->pressio_id);
    if (!comp) {
        fprintf(stderr, "no such compressor: %s (%s)\n", c->name, pressio_error_msg(library));
        return NULL;
    }
    char json[256];
    bench_compressor_opts_json(c, d, json, sizeof(json));

    /* Phase 1: JSON-serializable options (matches connector replay). */
    struct pressio_options *opts = pressio_options_new_json(library, json);
    if (pressio_compressor_set_options(comp, opts) != 0)
        fprintf(stderr, "%s set_options: %s\n", c->name, pressio_compressor_error_msg(comp));
    pressio_options_free(opts);

#ifdef BENCH_TIMING_WITH_CUDA
    /* Phase 2: userptrs CANNOT survive JSON -- re-attach the stream separately. */
    if (c->kind == BENCH_GPU_CODEC && c->stream_opt_key) {
        struct pressio_options *so = pressio_options_new();
        pressio_options_set_userptr(so, c->stream_opt_key, (void *)stream);
        pressio_compressor_set_options(comp, so);
        pressio_options_free(so);
    }
#endif
    return comp;
}

static int run_one(struct pressio *library, const bench_compressor_t *c,
                   const bench_dataset_t *d, const char *payload_path, FILE *csv) {
    void *hbuf = NULL;
    if (!load_field(d, &hbuf)) { fprintf(stderr, "skip %s (load failed)\n", d->name); return -1; }

    enum pressio_dtype pt = bench_dataset_pressio_dtype(d);
    size_t pdims[BENCH_MAX_RANK];
    bench_dataset_pressio_dims(d, pdims);            /* fastest-first for pressio */

    char wlabel[160], rlabel[160];
    snprintf(wlabel, sizeof(wlabel), "%s/%s/pressio/write", d->name, c->name);
    snprintf(rlabel, sizeof(rlabel), "%s/%s/pressio/read",  d->name, c->name);
    bench_report_t wr, rd;
    bench_report_reset(&wr, wlabel);
    bench_report_reset(&rd, rlabel);

#ifdef BENCH_TIMING_WITH_CUDA
    cudaStream_t stream = 0;
    if (c->kind == BENCH_GPU_CODEC) cudaStreamCreate(&stream);
    struct pressio_compressor *comp = make_compressor(library, c, d, stream);
#else
    struct pressio_compressor *comp = make_compressor(library, c, d);
#endif
    if (!comp) { free(hbuf); return -1; }

    struct pressio_data *input      = pressio_data_new_nonowning(pt, hbuf, d->rank, pdims);
    struct pressio_data *compressed = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
    struct pressio_data *output     = pressio_data_new_owning(pt, d->rank, pdims);

    /* ---------------- WRITE side: COMPRESS then IO ---------------- */
    double t0 = bench_now_ms();
#ifdef BENCH_TIMING_WITH_CUDA
    if (c->kind == BENCH_GPU_CODEC) {
        bench_gpu_timer_t g = bench_gpu_timer_create(stream);   /* same stream! */
        bench_gpu_timer_start(&g);
        pressio_compressor_compress(comp, input, compressed);   /* enqueues on stream */
        bench_gpu_timer_stop(&g);
        bench_report_add_gpu(&wr, BENCH_PHASE_COMPRESS, bench_gpu_timer_elapsed_ms(&g));
        bench_gpu_timer_destroy(&g);
    } else
#endif
    {
        bench_scope_t sc = bench_scope_begin(&wr, BENCH_PHASE_COMPRESS);
        pressio_compressor_compress(comp, input, compressed);   /* CPU codec */
        bench_scope_end(&sc);
    }

    size_t csize = 0;
    void  *cptr  = pressio_data_ptr(compressed, &csize);
    {   /* IO: write the compressed payload to disk */
        bench_scope_t si = bench_scope_begin(&wr, BENCH_PHASE_IO);
        FILE *pf = fopen(payload_path, "wb");
        if (pf) { fwrite(cptr, 1, csize, pf); fclose(pf); }
        bench_scope_end(&si);
    }
    wr.cpu_ms[BENCH_PHASE_TOTAL] = bench_now_ms() - t0;
    bench_report_derive_overhead(&wr);   /* ~0: proves the harness itself is thin */
    fprintf(stderr, "%-28s ratio=%.2fx\n", wlabel, (double)bench_num_bytes(d) / (double)csize);

    /* ---------------- READ side: IO then DECOMPRESS ---------------- */
    double r0 = bench_now_ms();
    struct pressio_data *reloaded = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
    {   /* IO: read the compressed payload back */
        bench_scope_t si = bench_scope_begin(&rd, BENCH_PHASE_IO);
        FILE *pf = fopen(payload_path, "rb");
        if (pf) {
            void *tmp = malloc(csize);
            size_t got = fread(tmp, 1, csize, pf);
            fclose(pf);
            pressio_data_free(reloaded);
            reloaded = pressio_data_new_move(pressio_byte_dtype, tmp, 1, &got,
                                             pressio_data_libc_free_fn, NULL);
        }
        bench_scope_end(&si);
    }
#ifdef BENCH_TIMING_WITH_CUDA
    if (c->kind == BENCH_GPU_CODEC) {
        bench_gpu_timer_t g = bench_gpu_timer_create(stream);
        bench_gpu_timer_start(&g);
        pressio_compressor_decompress(comp, reloaded, output);
        bench_gpu_timer_stop(&g);
        bench_report_add_gpu(&rd, BENCH_PHASE_COMPRESS, bench_gpu_timer_elapsed_ms(&g));
        bench_gpu_timer_destroy(&g);
    } else
#endif
    {
        bench_scope_t sc = bench_scope_begin(&rd, BENCH_PHASE_COMPRESS);
        pressio_compressor_decompress(comp, reloaded, output);
        bench_scope_end(&sc);
    }
    rd.cpu_ms[BENCH_PHASE_TOTAL] = bench_now_ms() - r0;
    bench_report_derive_overhead(&rd);

    bench_report_csv_row(csv, &wr);
    bench_report_csv_row(csv, &rd);

    pressio_data_free(input);
    pressio_data_free(compressed);
    pressio_data_free(output);
    pressio_data_free(reloaded);
    pressio_compressor_release(comp);
#ifdef BENCH_TIMING_WITH_CUDA
    if (c->kind == BENCH_GPU_CODEC) cudaStreamDestroy(stream);
#endif
    free(hbuf);
    return 0;
}

int main(int argc, char **argv) {
    const char *payload = (argc > 1) ? argv[1] : "payload.bin";
    const char *csvpath = (argc > 2) ? argv[2] : "results_pressio.csv";
    const char *only    = (argc > 3) ? argv[3] : NULL;  /* optional: one compressor */

    bench_datasets_validate();

    struct pressio *library = pressio_instance();
    FILE *csv = fopen(csvpath, "w");
    if (!csv) { perror("csv"); return 1; }
    bench_report_csv_header(csv);

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (access(d->path, R_OK) != 0) continue;
        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (only && __builtin_strcmp(only, c->name)) continue;
            run_one(library, c, d, payload, csv);
        }
    }
    fclose(csv);
    pressio_release(library);
    fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}