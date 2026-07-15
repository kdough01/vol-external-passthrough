/* ============================================================================
 * bench_timing.h  --  minimal timing helpers shared by all three harnesses
 * ----------------------------------------------------------------------------
 * CPU : std::chrono::steady_clock  (monotonic wall time, ~ns resolution)
 * GPU : CUDA events on the SAME stream the codec runs on (device time)
 * Out : one CSV row per (dataset, compressor, path, op, phase). All three
 *       harnesses share the schema, so the CSV files concatenate cleanly.
 *
 * This is the ONLY file that knows about clocks. Compile the harnesses as C++
 * (steady_clock). For GPU-event timing, define BENCH_TIMING_WITH_CUDA and build
 * the relevant harness with nvcc.
 * ==========================================================================*/
#ifndef BENCH_TIMING_H
#define BENCH_TIMING_H

#include <chrono>
#include <cstdio>

#ifdef BENCH_TIMING_WITH_CUDA
#include <cuda_runtime.h>
#endif

/* ---- CPU: steady_clock stopwatch ---------------------------------------- */
struct BenchCpuTimer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double stop_ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

/* ---- GPU: CUDA-event stopwatch on a given stream ------------------------- */
/* Records events on `stream`, so it captures exactly the work the codec issues
 * on that stream (cuszp is handed this same stream via cuszp:cuda_stream). Use
 * this instead of a CPU clock for GPU codecs: the compress call may return
 * before the kernel finishes, and stop_ms() synchronizes on the stop event. */
#ifdef BENCH_TIMING_WITH_CUDA
struct BenchGpuTimer {
    cudaEvent_t  a{}, b{};
    cudaStream_t stream{};
    explicit BenchGpuTimer(cudaStream_t s = 0) : stream(s) {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~BenchGpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a, stream); }
    double stop_ms() {
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);            /* wait for the recorded work */
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);    /* device time, milliseconds */
        return ms;
    }
};
#endif /* BENCH_TIMING_WITH_CUDA */

/* ---- shared CSV schema --------------------------------------------------- */
/* phase in {compress, io, total}; ratio filled on write rows, rmse on read.
 * ratio <= 0  => blank column;  rmse < 0 => blank column.                    */
static inline void bench_csv_header(FILE *f) {
    std::fprintf(f, "dataset,compressor,path,op,phase,ms,ratio,rmse\n");
    std::fflush(f);
}

static inline void bench_csv_row(FILE *f,
                                 const char *dataset, const char *compressor,
                                 const char *path, const char *op,
                                 const char *phase, double ms,
                                 double ratio, double rmse) {
    std::fprintf(f, "%s,%s,%s,%s,%s,%.4f,",
                 dataset, compressor, path, op, phase, ms);
    if (ratio > 0.0) std::fprintf(f, "%.4f,", ratio); else std::fprintf(f, ",");
    if (rmse  >= 0.0) std::fprintf(f, "%.6e", rmse);
    std::fprintf(f, "\n");
    std::fflush(f);
}

#endif /* BENCH_TIMING_H */