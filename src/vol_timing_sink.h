#ifndef VOL_TIMING_SINK_H
#define VOL_TIMING_SINK_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#ifdef __cplusplus
#include <chrono>
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double stage_ms;       /* host memcpy into ctx->stage_buf (partial writes)  */
    double compress_ms;    /* wall clock around the compress entry point        */
    double container_ms;   /* SET_EXTENT + dataspace mgmt + header assembly     */
    double io_ms;          /* H5VLdataset_write / _read calls                   */

    double pressio_call_ms;/* wall around pressio_compressor_compress()         */
    double device_ms;      /* CUDA event window (0.0 for CPU codecs)            */
    double transfer_ms;

    double total_ms;       /* wall clock, whole per-dataset loop body           */
    double residual_ms;    /* total - (stage+compress+container+io); want ~0    */
} vol_write_timing_t;

static inline double bench_now_ms(void) {
#ifdef __cplusplus
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);  /* steady_clock's source on Linux */
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
#endif
}

static inline FILE *vol_timing_csv(void) {
    static FILE *f = NULL;
    static int   tried = 0;
    if (!tried) {
        tried = 1;
        const char *path = getenv("BENCH_CSV");
        if (!path) path = "connector_timing.csv";
        int existed = (access(path, F_OK) == 0);
        f = fopen(path, "a");
        if (f && !existed)
            fprintf(f,
                "dataset,compressor,op,total_ms,"
                "stage_ms,compress_ms,container_ms,io_ms,"
                "pressio_call_ms,device_ms,transfer_ms,vol_compress_ms,pressio_host_ms,"
                "residual_ms,residual_frac,h2d_ms\n");
    }
    return f;
}

static inline void vol_timing_finalize(vol_write_timing_t *t,
                                       const char *dataset, const char *compressor,
                                       double tol_frac) {
    if (!t) return;
    t->residual_ms = t->total_ms - (t->stage_ms + t->compress_ms
                                  + t->container_ms + t->io_ms + t->h2d_ms);
    if (getenv("VOL_TIMING_STRICT") && t->total_ms > 0.0 &&
        fabs(t->residual_ms) > tol_frac * t->total_ms) {
        fprintf(stderr,
            "[VOL timing] UNACCOUNTED %.3f ms of %.3f ms (%.1f%%) "
            "dset=%s comp=%s -- a timer is missing\n",
            t->residual_ms, t->total_ms,
            100.0 * t->residual_ms / t->total_ms,
            dataset ? dataset : "?", compressor ? compressor : "?");
    }
}

/* Emit one fully-resolved row. */
static inline void vol_timing_emit_ex(const char *dataset, const char *compressor,
                                      const char *op,
                                      const vol_write_timing_t *t) {
    FILE *f = vol_timing_csv();
    if (!f || !t) return;

    double vol_compress = t->compress_ms - t->pressio_call_ms - t->transfer_ms;
    double pressio_host = t->pressio_call_ms - t->device_ms;
    double residual_frac = (t->total_ms > 0.0) ? t->residual_ms / t->total_ms : 0.0;

    fprintf(f, "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.3f\n",
            dataset ? dataset : "?",
            compressor ? compressor : "none",
            op ? op : "?",
            t->total_ms,
            t->stage_ms, t->compress_ms, t->container_ms, t->io_ms,
            t->pressio_call_ms, t->device_ms, t->transfer_ms,
            vol_compress, pressio_host,
            t->residual_ms, residual_frac,t->h2d_ms);
    fflush(f);
}

static inline void vol_timing_emit(const char *dataset, const char *compressor,
                                   const char *op, double total_ms,
                                   double compress_ms, double io_ms) {
    vol_write_timing_t t;
    memset(&t, 0, sizeof(t));
    t.total_ms    = total_ms;
    t.compress_ms = compress_ms;
    t.io_ms       = io_ms;
    vol_timing_finalize(&t, dataset, compressor, 1.0);  /* never warn here */
    vol_timing_emit_ex(dataset, compressor, op, &t);
}

#ifdef __cplusplus
}
#endif

#endif /* VOL_TIMING_SINK_H */