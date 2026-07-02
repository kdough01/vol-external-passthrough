#ifndef BENCH_CONFIG_H
#define BENCH_CONFIG_H

//  H5Dwrite_ms - compress_ms   ~=  write-side HDF5 + I/O overhead
//  H5Dread_ms  - decompress_ms ~=  read-side  HDF5 + I/O overhead

#include <stddef.h>

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

#endif /* BENCH_CONFIG_H */