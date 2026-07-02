// Times RAW libpressio compress/decompress

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <libpressio/libpressio.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "bench_config.h"

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

static enum pressio_dtype dt_pressio(bench_dtype_t t) {
    return (t == DT_F32) ? pressio_float_dtype : pressio_double_dtype;
}

static void *read_raw(const char *path, size_t nelem, bench_dtype_t t) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return NULL; }
    size_t es = bench_dt_size(t);
    void *buf = malloc(nelem * es);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, es, nelem, f);
    fclose(f);
    if (got != nelem) {
        fprintf(stderr, "  short read %s: got %zu, expected %zu\n", path, got, nelem);
        free(buf); return NULL;
    }
    return buf;
}

static void err_metrics(const void *a, const void *b, size_t n, bench_dtype_t t,
                        double *maxabs, double *rmse) {
    double m = 0.0, s = 0.0;
    if (t == DT_F32) {
        const float *x = a, *y = b;
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)x[i] - (double)y[i]);
            if (d > m) m = d;
            s += d * d;
        }
    } else {
        const double *x = a, *y = b;
        for (size_t i = 0; i < n; i++) {
            double d = fabs(x[i] - y[i]);
            if (d > m) m = d;
            s += d * d;
        }
    }
    *maxabs = m;
    *rmse   = sqrt(s / (double)n);
}

int main(void) {
    struct pressio *library = pressio_instance();

#ifdef USE_CUDA
    cudaStream_t stream = 0;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "  warning: cudaStreamCreate failed; GPU compressors may error\n");
        stream = 0;
    }
#endif

    printf("# libpressio-only timing (no HDF5, no VOL)\n");
    printf("# %-20s %-8s %12s %12s %8s %12s %12s\n",
           "dataset", "comp", "compress_ms", "decompress_ms", "ratio", "max_abs", "rmse");
    fflush(stdout);

    for (int d = 0; d < BENCH_N_DATASETS; d++) {
        const bench_dataset_t *ds = &BENCH_DATASETS[d];
        size_t nelem = ds->nx * ds->ny * ds->nz;
        size_t esize = bench_dt_size(ds->dtype);
        enum pressio_dtype pdt = dt_pressio(ds->dtype);
        size_t pdims[3] = { ds->nx, ds->ny, ds->nz };

        void *field = read_raw(ds->path, nelem, ds->dtype);
        if (!field) { fprintf(stderr, "  skipping %s (read failed)\n", ds->name); continue; }

        for (int c = 0; c < BENCH_N_COMPRESSORS; c++) {
            const bench_comp_t *cc = &BENCH_COMPRESSORS[c];

#ifndef USE_CUDA
            if (cc->is_gpu) {
                printf("  %-20s %-8s   (skipped: built without CUDA)\n", ds->name, cc->label);
                fflush(stdout);
                continue;
            }
#endif
            struct pressio_compressor *comp = pressio_get_compressor(library, cc->pressio_id);
            if (!comp) {
                printf("  %-20s %-8s   (compressor not registered)\n", ds->name, cc->label);
                fflush(stdout);
                continue;
            }

            /* Options: mirror the JSON used on the VOL side, key-by-key. */
            struct pressio_options *opts = pressio_compressor_get_options(comp);
            if (cc->abs >= 0.0)   pressio_options_set_double(opts, "pressio:abs", cc->abs);
            if (cc->mode_str)     pressio_options_set_string(opts, "cuszp:mode_str", cc->mode_str);
#ifdef USE_CUDA
            if (cc->is_gpu && stream) {
                /* cudaStream_t is an opaque pointer; hand it over as a userptr.
                 * Adjust the key if your build expects a different one. */
                pressio_options_set_userptr(opts, "cuszp:stream", (void *)stream);
            }
#endif
            if (pressio_compressor_set_options(comp, opts) != 0) {
                printf("  %-20s %-8s   (set_options failed: %s)\n",
                       ds->name, cc->label, pressio_compressor_error_msg(comp));
                pressio_options_free(opts);
                pressio_compressor_release(comp);
                fflush(stdout);
                continue;
            }
            pressio_options_free(opts);

            /* Host input; libpressio's domain manager migrates host->device
             * internally for GPU compressors. */
            struct pressio_data *input =
                pressio_data_new_nonowning(pdt, field, 3, pdims);
            struct pressio_data *compressed =
                pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
            struct pressio_data *decompressed =
                pressio_data_new_owning(pdt, 3, pdims);

            struct timespec t0, t1;

            clock_gettime(CLOCK_MONOTONIC, &t0);
            int cret = pressio_compressor_compress(comp, input, compressed);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double cms = elapsed_ms(t0, t1);

            if (cret != 0) {
                printf("  %-20s %-8s   (compress failed: %s)\n",
                       ds->name, cc->label, pressio_compressor_error_msg(comp));
                pressio_data_free(input);
                pressio_data_free(compressed);
                pressio_data_free(decompressed);
                pressio_compressor_release(comp);
                fflush(stdout);
                continue;
            }

            size_t comp_bytes = 0;
            (void)pressio_data_ptr(compressed, &comp_bytes);

            clock_gettime(CLOCK_MONOTONIC, &t0);
            int dret = pressio_compressor_decompress(comp, compressed, decompressed);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dms = elapsed_ms(t0, t1);

            double maxabs = 0.0, rmse = 0.0;
            if (dret == 0) {
                size_t out_bytes = 0;
                void *out = pressio_data_ptr(decompressed, &out_bytes);
                if (out && out_bytes >= nelem * esize)
                    err_metrics(field, out, nelem, ds->dtype, &maxabs, &rmse);
            }

            double ratio = (comp_bytes > 0)
                         ? (double)(nelem * esize) / (double)comp_bytes : 0.0;

            if (dret != 0) {
                printf("  %-20s %-8s %12.2f %12s %8.2f %12s %12s  (decompress failed: %s)\n",
                       ds->name, cc->label, cms, "-", ratio, "-", "-",
                       pressio_compressor_error_msg(comp));
            } else {
                printf("  %-20s %-8s %12.2f %12.2f %8.2f %12.3e %12.3e\n",
                       ds->name, cc->label, cms, dms, ratio, maxabs, rmse);
            }
            fflush(stdout);

            pressio_data_free(input);
            pressio_data_free(compressed);
            pressio_data_free(decompressed);
            pressio_compressor_release(comp);
        }

        free(field);
    }

#ifdef USE_CUDA
    if (stream) cudaStreamDestroy(stream);
#endif
    pressio_release(library);
    return 0;
}