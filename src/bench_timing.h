#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>
#include <string.h>
#include <stdio.h>

#ifdef BENCH_TIMING_WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline double bench_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec * 1.0e-6;
}

typedef enum {
    BENCH_PHASE_TOTAL    = 0, /* end-to-end H5Dwrite / H5Dread wall time     */
    BENCH_PHASE_COMPRESS = 1, /* (de)compressor only: CPU codec or GPU codec */
    BENCH_PHASE_IO       = 2, /* bytes actually moved to/from the file       */
    BENCH_PHASE_OVERHEAD = 3, /* VOL bookkeeping: attrs, metadata, buffers   */
    BENCH_PHASE_COUNT    = 4
} bench_phase_t;

static inline const char *bench_phase_name(bench_phase_t p) {
    switch (p) {
        case BENCH_PHASE_TOTAL:    return "total";
        case BENCH_PHASE_COMPRESS: return "compress";
        case BENCH_PHASE_IO:       return "io";
        case BENCH_PHASE_OVERHEAD: return "overhead";
        default:                   return "?";
    }
}

typedef struct {
    double        cpu_ms[BENCH_PHASE_COUNT];  /* accumulated host wall time  */
    double        gpu_ms[BENCH_PHASE_COUNT];  /* accumulated device time     */
    unsigned long calls [BENCH_PHASE_COUNT];  /* #intervals folded in        */
    const char   *label;                      /* e.g. "miranda/cuszp/write"  */
} bench_report_t;

static inline void bench_report_reset(bench_report_t *r, const char *label) {
    memset(r, 0, sizeof(*r));
    r->label = label;
}

static inline void bench_report_add_cpu(bench_report_t *r,
                                        bench_phase_t p, double ms) {
    if (!r) return;
    r->cpu_ms[p] += ms;
    r->calls[p]  += 1;
}

static inline void bench_report_add_gpu(bench_report_t *r,
                                        bench_phase_t p, double ms) {
    if (!r) return;
    r->gpu_ms[p] += ms;
}

static inline void bench_report_derive_overhead(bench_report_t *r) {
    double explained = r->cpu_ms[BENCH_PHASE_COMPRESS] + r->cpu_ms[BENCH_PHASE_IO];
    double oh = r->cpu_ms[BENCH_PHASE_TOTAL] - explained;
    r->cpu_ms[BENCH_PHASE_OVERHEAD] = (oh > 0.0) ? oh : 0.0;
}

static inline void bench_report_csv_header(FILE *f) {
    fprintf(f, "label,total_ms,compress_cpu_ms,compress_gpu_ms,io_ms,overhead_ms,"
               "overhead_pct,compress_calls\n");
}
static inline void bench_report_csv_row(FILE *f, const bench_report_t *r) {
    double total = r->cpu_ms[BENCH_PHASE_TOTAL];
    double oh    = r->cpu_ms[BENCH_PHASE_OVERHEAD];
    fprintf(f, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%lu\n",
            r->label ? r->label : "",
            total,
            r->cpu_ms[BENCH_PHASE_COMPRESS],
            r->gpu_ms[BENCH_PHASE_COMPRESS],
            r->cpu_ms[BENCH_PHASE_IO],
            oh,
            (total > 0.0) ? 100.0 * oh / total : 0.0,
            r->calls[BENCH_PHASE_COMPRESS]);
}

typedef struct {
    bench_report_t *report;
    bench_phase_t   phase;
    double          t0_ms;
} bench_scope_t;

static inline bench_scope_t bench_scope_begin(bench_report_t *r, bench_phase_t p) {
    bench_scope_t s;
    s.report = r;
    s.phase  = p;
    s.t0_ms  = bench_now_ms();
    return s;
}
static inline double bench_scope_end(bench_scope_t *s) {
    double dt = bench_now_ms() - s->t0_ms;
    bench_report_add_cpu(s->report, s->phase, dt);
    return dt;
}

#if defined(__cplusplus)
  #define BENCH_TLS thread_local
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define BENCH_TLS _Thread_local
#else
  #define BENCH_TLS __thread
#endif

static BENCH_TLS bench_report_t *bench_tls_current = NULL;

static inline void bench_tls_set(bench_report_t *r) { bench_tls_current = r; }
static inline bench_report_t *bench_tls_get(void)   { return bench_tls_current; }

#ifdef BENCH_TIMING_WITH_CUDA

#define BENCH_CUDA_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[bench_timing] CUDA error %s at %s:%d: %s\n",   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_e));      \
        }                                                                     \
    } while (0)

typedef struct {
    cudaEvent_t  start;
    cudaEvent_t  stop;
    cudaStream_t stream;   /* the compressor's stream (NOT the default 0)   */
    int          active;
} bench_gpu_timer_t;

static inline bench_gpu_timer_t bench_gpu_timer_create(cudaStream_t stream) {
    bench_gpu_timer_t t;
    t.stream = stream;
    t.active = 0;
    BENCH_CUDA_CHECK(cudaEventCreate(&t.start));
    BENCH_CUDA_CHECK(cudaEventCreate(&t.stop));
    return t;
}

static inline void bench_gpu_timer_start(bench_gpu_timer_t *t) {
    BENCH_CUDA_CHECK(cudaEventRecord(t->start, t->stream));
    t->active = 1;
}

static inline void bench_gpu_timer_stop(bench_gpu_timer_t *t) {
    BENCH_CUDA_CHECK(cudaEventRecord(t->stop, t->stream));
}

static inline float bench_gpu_timer_elapsed_ms(bench_gpu_timer_t *t) {
    float ms = 0.0f;
    BENCH_CUDA_CHECK(cudaEventSynchronize(t->stop));
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    t->active = 0;
    return ms;
}

static inline void bench_gpu_timer_destroy(bench_gpu_timer_t *t) {
    BENCH_CUDA_CHECK(cudaEventDestroy(t->start));
    BENCH_CUDA_CHECK(cudaEventDestroy(t->stop));
}

#endif /* BENCH_TIMING_WITH_CUDA */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BENCH_TIMING_H */