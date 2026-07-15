/* ============================================================================
 * bench_pressio_timing.cc  --  APPROACH 1 of 3: libpressio DIRECT (no HDF5)
 * ----------------------------------------------------------------------------
 * The lower-bound comparator: same codec config as the VOL and the filter, but
 * zero HDF5/VOL machinery. Compress in memory, fwrite the payload; read it back,
 * decompress. This harness sees ALL three phases cleanly:
 *     write: compress + io          read: io + decompress
 * CPU timing via std::chrono::steady_clock; GPU (cuszp) via CUDA events on the
 * codec's stream. TOTAL is wall time (steady_clock); for GPU codecs TOTAL will
 * not exactly equal compress+io because the compress phase is device time.
 *
 * Build (CPU only):
 *   g++  -O2 -std=c++17 bench_pressio_timing.cc -lpressio -o bench_pressio_timing
 * Build (GPU-event timing for cuszp):
 *   nvcc -O2 -std=c++17 -x cu -DBENCH_TIMING_WITH_CUDA bench_pressio_timing.cc \
 *        -lpressio -lcudart -o bench_pressio_timing
 * ==========================================================================*/
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include <libpressio.h>
#ifdef BENCH_TIMING_WITH_CUDA
#include <cuda_runtime.h>
#endif

/* Some libpressio builds do NOT declare the C JSON options builder in the
 * public header -> implicit-int return truncates the 64-bit pointer -> segfault.
 * Declare it explicitly. Confirm name/signature for your build with:
 *     grep -rn "options.*json\|from_json" $LP_VIEW/include                     */
extern "C" struct pressio_options *
pressio_options_new_json(struct pressio *library, const char *json);

#define BENCH_CONFIG_ENABLE_PRESSIO
#include "bench_config.h"
#include "bench_timing.h"

/* Load one raw field into a host buffer. */
static size_t load_field(const bench_dataset_t *d, void **out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = std::malloc(nbytes);
    if (!buf) return 0;
    FILE *f = std::fopen(d->path, "rb");
    if (!f) { std::free(buf); return 0; }
    size_t got = std::fread(buf, 1, nbytes, f);
    std::fclose(f);
    if (got != nbytes) { std::free(buf); return 0; }
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
    const char *pid = (c->pressio_id && c->pressio_id[0]) ? c->pressio_id : "noop";
    struct pressio_compressor *comp = pressio_get_compressor(library, pid);
    if (!comp) {
        std::fprintf(stderr, "no such compressor: %s (id=%s): %s\n",
                     c->name, pid, pressio_error_msg(library));
        return NULL;
    }
    char json[256];
    bench_compressor_opts_json(c, d, json, sizeof(json));

    struct pressio_options *opts = pressio_options_new_json(library, json);
    if (!opts) {
        std::fprintf(stderr, "%s: JSON parse failed for '%s'\n", c->name, json);
        pressio_compressor_release(comp);
        return NULL;
    }
    if (pressio_compressor_set_options(comp, opts) != 0)
        std::fprintf(stderr, "%s set_options: %s\n",
                     c->name, pressio_compressor_error_msg(comp));
    pressio_options_free(opts);

#ifdef BENCH_TIMING_WITH_CUDA
    /* userptrs CANNOT survive JSON -- re-attach the stream separately. */
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
    if (!load_field(d, &hbuf)) {
        std::fprintf(stderr, "skip %s (load failed)\n", d->name);
        return -1;
    }

    enum pressio_dtype pt = bench_dataset_pressio_dtype(d);
    size_t pdims[BENCH_MAX_RANK];
    bench_dataset_pressio_dims(d, pdims);

    const char *pid = (c->pressio_id && c->pressio_id[0]) ? c->pressio_id : "noop";
    int is_noop = (std::strcmp(pid, "noop") == 0);
    int is_gpu  = (c->kind == BENCH_GPU_CODEC);

    size_t raw = bench_num_bytes(d);

#ifdef BENCH_TIMING_WITH_CUDA
    cudaStream_t stream = 0;
    if (is_gpu) cudaStreamCreate(&stream);
    struct pressio_compressor *comp = make_compressor(library, c, d, stream);
#else
    struct pressio_compressor *comp = make_compressor(library, c, d);
#endif
    if (!comp) {
        std::free(hbuf);
#ifdef BENCH_TIMING_WITH_CUDA
        if (is_gpu) cudaStreamDestroy(stream);
#endif
        return -1;
    }

    struct pressio_data *input =
        pressio_data_new_nonowning(pt, hbuf, d->rank, pdims);
    struct pressio_data *compressed =
        pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    /* ---------------- WRITE: compress, then io ---------------- */
    BenchCpuTimer wtotal; wtotal.start();

    int cerr = 0;
    double compress_ms = 0.0;
#ifdef BENCH_TIMING_WITH_CUDA
    if (is_gpu) {
        BenchGpuTimer g(stream);            /* CUDA events on the codec stream */
        g.start();
        cerr = pressio_compressor_compress(comp, input, compressed);
        compress_ms = g.stop_ms();
    } else
#endif
    {
        BenchCpuTimer t; t.start();
        cerr = pressio_compressor_compress(comp, input, compressed);
        compress_ms = t.stop_ms();
    }
    if (cerr)
        std::fprintf(stderr, "%s/%s compress failed: %s\n",
                     d->name, c->name, pressio_compressor_error_msg(comp));

    size_t csize = 0;
    void  *cptr  = pressio_data_ptr(compressed, &csize);
    if (!cerr && cptr && csize > 0) {
        double ratio = (double)raw / (double)csize;

        BenchCpuTimer io; io.start();
        FILE *pf = std::fopen(payload_path, "wb");
        if (pf) { std::fwrite(cptr, 1, csize, pf); std::fclose(pf); }
        double io_ms = io.stop_ms();

        double total_ms = wtotal.stop_ms();

        bench_csv_row(csv, d->name, c->name, "pressio", "write", "compress",
                      compress_ms, ratio, -1.0);
        bench_csv_row(csv, d->name, c->name, "pressio", "write", "io",
                      io_ms, ratio, -1.0);
        bench_csv_row(csv, d->name, c->name, "pressio", "write", "total",
                      total_ms, ratio, -1.0);
        std::fprintf(stderr, "%-24s W compress=%8.2f io=%8.2f total=%8.2f ratio=%6.2fx\n",
                     (std::string(d->name) + "/" + c->name).c_str(),
                     compress_ms, io_ms, total_ms, ratio);
    } else {
        std::fprintf(stderr, "%s/%s: no compressed output, skipping\n",
                     d->name, c->name);
    }

    /* Free field + input before the read side: keeps peak memory to ~1x field. */
    pressio_data_free(input);
    std::free(hbuf);
    pressio_data_free(compressed);   /* payload is on disk now */

    /* ---------------- READ: io, then decompress ---------------- */
    if (!cerr && csize > 0) {
        BenchCpuTimer rtotal; rtotal.start();

        struct pressio_data *reloaded =
            pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

        BenchCpuTimer io; io.start();
        FILE *pf = std::fopen(payload_path, "rb");
        if (pf) {
            void *tmp = std::malloc(csize);
            size_t got = std::fread(tmp, 1, csize, pf);
            std::fclose(pf);
            pressio_data_free(reloaded);
            reloaded = pressio_data_new_move(pressio_byte_dtype, tmp, 1, &got,
                                             pressio_data_libc_free_fn, NULL);
        }
        double io_ms = io.stop_ms();

        double decompress_ms = 0.0;
        if (!is_noop) {   /* real codec: allocate output JIT, decompress */
            struct pressio_data *output = pressio_data_new_owning(pt, d->rank, pdims);
            int derr = 0;
#ifdef BENCH_TIMING_WITH_CUDA
            if (is_gpu) {
                BenchGpuTimer g(stream);
                g.start();
                derr = pressio_compressor_decompress(comp, reloaded, output);
                decompress_ms = g.stop_ms();
            } else
#endif
            {
                BenchCpuTimer t; t.start();
                derr = pressio_compressor_decompress(comp, reloaded, output);
                decompress_ms = t.stop_ms();
            }
            if (derr)
                std::fprintf(stderr, "%s/%s decompress failed: %s\n",
                             d->name, c->name, pressio_compressor_error_msg(comp));
            pressio_data_free(output);
        }
        /* noop: nothing to decompress; compress phase stays 0. */

        double total_ms = rtotal.stop_ms();

        bench_csv_row(csv, d->name, c->name, "pressio", "read", "io",
                      io_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "pressio", "read", "compress",
                      decompress_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "pressio", "read", "total",
                      total_ms, -1.0, -1.0);

        pressio_data_free(reloaded);
    }

    pressio_compressor_release(comp);
#ifdef BENCH_TIMING_WITH_CUDA
    if (is_gpu) cudaStreamDestroy(stream);
#endif
    return 0;
}

int main(int argc, char **argv) {
    const char *payload = (argc > 1) ? argv[1] : "payload.bin";
    const char *csvpath = (argc > 2) ? argv[2] : "results_pressio.csv";
    const char *only    = (argc > 3) ? argv[3] : NULL;  /* optional: one compressor */

    bench_datasets_validate();

    struct pressio *library = pressio_instance();
    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (access(d->path, R_OK) != 0) continue;
        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (only && std::strcmp(only, c->name)) continue;
            run_one(library, c, d, payload, csv);
        }
    }
    std::fclose(csv);
    pressio_release(library);
    std::fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}