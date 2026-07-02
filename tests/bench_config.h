#ifndef BENCH_CONFIG_H
#define BENCH_CONFIG_H

/* ============================================================================
 *  Shared benchmark configuration + error metric.
 *
 *  Both bench_vol_timing.c (through the passthrough VOL) and
 *  bench_pressio_timing.c (raw libpressio, no HDF5) include THIS file, so the
 *  two runs exercise an identical matrix of datasets x compressors and score
 *  fidelity the same way. That is what makes the timing subtraction meaningful:
 *
 *      H5Dwrite_ms - compress_ms   ~=  write-side HDF5 + I/O overhead
 *      H5Dread_ms  - decompress_ms ~=  read-side  HDF5 + I/O overhead
 *
 *  Edit the two tables below to add SDRBench fields or compressors.
 * ========================================================================== */

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
    double      abs;
    const char *mode_str;
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
    { "noop",  "noop",  -1.0,  NULL,      0, NULL },
    { "bzip2", "bzip2", -1.0,  NULL,      0, NULL },
    { "sz3",   "sz3",   1e-3,  NULL,      0, "{\"pressio:abs\": 1e-3}" },
    { "cuszp", "cuszp", 1e-3,  "outlier", 1,
      "{\"pressio:abs\": 1e-3, \"cuszp:mode_str\": \"outlier\"}" },

    /* { "nvcomp", "nvcomp", -1.0, NULL, 1, NULL }, */
    /* { "cusz",   "cusz",   1e-3, NULL, 1, "{\"pressio:abs\": 1e-3}" }, */
};
#define BENCH_N_COMPRESSORS \
    ((int)(sizeof(BENCH_COMPRESSORS) / sizeof(BENCH_COMPRESSORS[0])))

static inline size_t bench_dt_size(bench_dtype_t t) {
    return (t == DT_F32) ? sizeof(float) : sizeof(double);
}

static inline void
bench_err_metrics(const void *a, const void *b, size_t n, bench_dtype_t t,
                  double *maxabs, double *rmse,
                  size_t *n_special, size_t *n_bad)
{
    double m = 0.0, s = 0.0;
    size_t cmp = 0, special = 0, bad = 0;

    for (size_t i = 0; i < n; i++) {
        double xv, yv;
        if (t == DT_F32) { xv = ((const float  *)a)[i]; yv = ((const float  *)b)[i]; }
        else             { xv = ((const double *)a)[i]; yv = ((const double *)b)[i]; }

        if (!isfinite(xv)) { special++; continue; }
        if (!isfinite(yv)) { bad++;     continue; }

        double d = fabs(xv - yv);
        if (d > m) m = d;
        s += d * d;
        cmp++;
    }

    *maxabs    = m;
    *rmse      = cmp ? sqrt(s / (double)cmp) : 0.0;
    *n_special = special;
    *n_bad     = bad;
}

#endif /* BENCH_CONFIG_H */