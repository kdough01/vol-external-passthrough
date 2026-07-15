#ifndef VOL_TIMING_SINK_H
#define VOL_TIMING_SINK_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>   /* access */

/* Clock headers must be included OUTSIDE extern "C": wrapping <chrono> in
 * extern "C" gives its templates C linkage and fails to compile. */
#ifdef __cplusplus
#include <chrono>
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

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

/* Opened once, appended thereafter. Header written only if the file is new. */
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
            fprintf(f, "dataset,compressor,op,total_ms,compress_ms,io_ms,"
                       "overhead_ms,overhead_pct\n");
    }
    return f;
}

/* Emit one phase-breakdown row. overhead = total - compress - io (clamped). */
static inline void vol_timing_emit(const char *dataset, const char *compressor,
                                   const char *op, double total_ms,
                                   double compress_ms, double io_ms) {
    FILE *f = vol_timing_csv();
    if (!f) return;
    double oh = total_ms - compress_ms - io_ms;
    if (oh < 0.0) oh = 0.0;
    fprintf(f, "%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.2f\n",
            dataset ? dataset : "?",
            compressor ? compressor : "none",
            op, total_ms, compress_ms, io_ms, oh,
            total_ms > 0.0 ? 100.0 * oh / total_ms : 0.0);
    fflush(f);   /* survive a mid-run crash */
}

#ifdef __cplusplus
}
#endif

#endif /* VOL_TIMING_SINK_H */