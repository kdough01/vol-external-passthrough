#ifndef BENCH_CONFIG_H
#define BENCH_CONFIG_H

#include <stddef.h>
#include <math.h>

typedef enum { DT_F32 = 0, DT_F64 = 1 } bench_dtype_t;

typedef struct {
    const char   *name;
    const char   *path;
    size_t        nx, ny, nz;
    bench_dtype_t dtype;
} bench_dataset_t;

typedef struct {
    const char *label;
    const char *pressio_id;
    int         is_gpu;
    const char *opts_json;
} bench_comp_t;

static const bench_dataset_t BENCH_DATASETS[] = {
    { "hurricane_CLOUDf01",
      "/lcrc/project/ECP-EZ/public/compression/Hurricane-ISABEL/nonclean-data/CLOUDf01.bin",
      100, 500, 500, DT_F32 },

    { "miranda_density",
      "/lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384/density.f64",
      256, 384, 384, DT_F64 },

};
#define BENCH_N_DATASETS ((int)(sizeof(BENCH_DATASETS) / sizeof(BENCH_DATASETS[0])))

static const bench_comp_t BENCH_COMPRESSORS[] = {
    { "noop",  "noop",  0, NULL },
    { "bzip2", "bzip2", 0, "{\"bzip2:block_size\": 9}" },
    { "sz3",   "sz3",   0,
      "{\"sz3:error_bound_mode_str\": \"abs\", \"sz3:abs_error_bound\": 1e-3}" },
    { "cuszp", "cuszp", 1,
      "{\"pressio:abs\": 1e-3, \"cuszp:mode_str\": \"outlier\"}" },

    /* { "nvcomp", "nvcomp", 1, NULL }, */
    /* { "cusz",   "cusz",   1, "{\"pressio:abs\": 1e-3}" }, */
};
#define BENCH_N_COMPRESSORS \
    ((int)(sizeof(BENCH_COMPRESSORS) / sizeof(BENCH_COMPRESSORS[0])))

static inline size_t bench_dt_size(bench_dtype_t t) {
    return (t == DT_F32) ? sizeof(float) : sizeof(double);
}

typedef struct { double min, max, mean, rmse; } Stats;

static inline Stats
bench_compute_stats(const void *orig, const void *decomp, size_t n, bench_dtype_t t)
{
    double o0 = (t == DT_F32) ? (double)((const float  *)orig)[0]
                              :         ((const double *)orig)[0];
    Stats s = { o0, o0, 0.0, 0.0 };
    double sum = 0.0, sse = 0.0;

    for (size_t i = 0; i < n; i++) {
        double ov, dv;
        if (t == DT_F32) { ov = ((const float  *)orig)[i]; dv = ((const float  *)decomp)[i]; }
        else             { ov = ((const double *)orig)[i]; dv = ((const double *)decomp)[i]; }

        if (ov < s.min) s.min = ov;
        if (ov > s.max) s.max = ov;
        sum += ov;
        double diff = ov - dv;
        sse += diff * diff;
    }

    s.mean = sum / (double)n;
    s.rmse = sqrt(sse / (double)n);
    return s;
}

#endif /* BENCH_CONFIG_H */