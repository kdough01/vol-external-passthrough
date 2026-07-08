#ifndef BENCH_DATASETS_H
#define BENCH_DATASETS_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BENCH_DATA_ROOT
#define BENCH_DATA_ROOT "/lcrc/project/ECP-EZ/public/compression"
#endif

#define BENCH_MAX_RANK 4

typedef enum {
    BENCH_F32 = 0,   /* IEEE single precision, little-endian (.f32) */
    BENCH_F64 = 1    /* IEEE double precision, little-endian (.f64/.d64) */
} bench_dtype_t;

typedef enum {
    BENCH_BOUND_ABS = 0,   /* pointwise absolute error bound */
    BENCH_BOUND_REL = 1    /* value-range relative error bound */
} bench_bound_mode_t;

typedef struct {
    const char        *name;        /* short id used in output filenames/logs */
    const char        *path;        /* raw binary field on disk (one field)   */
    int                rank;        /* number of dimensions                   */
    size_t             dims[BENCH_MAX_RANK]; /* row-major (HDF5 order)         */
    bench_dtype_t      dtype;
    bench_bound_mode_t bound_mode;  /* default error control for lossy runs   */
    double             bound;       /* default error bound value              */
    int                gpu_suitable;/* 1 = large/contiguous enough for GPU codec */
    const char        *note;        /* provenance / caveats                   */
} bench_dataset_t;

static const bench_dataset_t BENCH_DATASETS[] = {
    {
        "miranda",
        BENCH_DATA_ROOT "/Miranda/SDRBENCH-Miranda-256x384x384/density.f64",
        3, {256, 384, 384, 0}, BENCH_F64,
        BENCH_BOUND_REL, 1e-3, 1,
        "Clean f64; cuszp returns RMSE~2.7e-4 here. SDRBench names it density.d64."
    },

    {
        "hurricane",
        BENCH_DATA_ROOT "/Hurricane-ISABEL/Pf48.bin.f32",
        3, {100, 500, 500, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 1,
        "Use a CLEARED field. nonclean-data fields contain NaN fill -> bad for fidelity."
    },

    {
        "nyx",
        BENCH_DATA_ROOT "/NYX_baryon_density_512/baryon_density.f32",
        3, {512, 512, 512, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 1,
        "Confirm field name; SDRBench canonical also ships temperature.f32 etc."
    },

    {
        "s3d",
        BENCH_DATA_ROOT "/S3D/stat_planar.1.1000E-03.field.d64",
        3, {500, 500, 500, 0}, BENCH_F64,
        BENCH_BOUND_REL, 1e-3, 1,
        "SDRBench lists S3D as f64 (.d64) despite one stray f32 line on the site."
    },

    {
        "cesm_atm_2d",
        BENCH_DATA_ROOT "/cesm/CLDHGH_1_1800_3600.f32",
        2, {1800, 3600, 0, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-2, 0,
        "2D f32. Cluster may hold only the 26x1800x3600 3D version -- confirm path."
    },

    {
        "scale_letkf",
        BENCH_DATA_ROOT "/scale-letkf/PRES-98x1200x1200.f32",
        3, {98, 1200, 1200, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 1,
        "Confirm which variable/field file is present (T-, PRES-, U-, ...)."
    },
};

#define BENCH_NUM_DATASETS \
    ((int)(sizeof(BENCH_DATASETS) / sizeof(BENCH_DATASETS[0])))

static inline size_t bench_dtype_size(bench_dtype_t t) {
    return (t == BENCH_F64) ? 8u : 4u;
}

static inline const char *bench_dtype_name(bench_dtype_t t) {
    return (t == BENCH_F64) ? "double" : "float";  /* libpressio dtype string */
}

static inline size_t bench_num_elements(const bench_dataset_t *d) {
    size_t n = 1;
    for (int i = 0; i < d->rank; ++i) n *= d->dims[i];
    return n;
}

static inline size_t bench_num_bytes(const bench_dataset_t *d) {
    return bench_num_elements(d) * bench_dtype_size(d->dtype);
}

static inline const char *bench_bound_mode_name(bench_bound_mode_t m) {
    return (m == BENCH_BOUND_ABS) ? "abs" : "rel";
}

static inline const bench_dataset_t *bench_dataset_by_name(const char *name) {
    for (int i = 0; i < BENCH_NUM_DATASETS; ++i)
        if (name && BENCH_DATASETS[i].name &&
            0 == __builtin_strcmp(name, BENCH_DATASETS[i].name))
            return &BENCH_DATASETS[i];
    return NULL;
}

static inline int bench_datasets_validate(void) {
    int found = 0;
    fprintf(stderr, "%-14s %-6s %-4s %-22s %-10s %s\n",
            "name", "dtype", "rank", "dims", "MiB", "exists");
    for (int i = 0; i < BENCH_NUM_DATASETS; ++i) {
        const bench_dataset_t *d = &BENCH_DATASETS[i];
        char dims[64]; size_t off = 0;
        for (int k = 0; k < d->rank; ++k)
            off += (size_t)snprintf(dims + off, sizeof(dims) - off,
                                    k ? "x%zu" : "%zu", d->dims[k]);
        int ok = (access(d->path, R_OK) == 0);
        found += ok;
        fprintf(stderr, "%-14s %-6s %-4d %-22s %-10.1f %s\n",
                d->name, bench_dtype_name(d->dtype), d->rank, dims,
                bench_num_bytes(d) / (1024.0 * 1024.0),
                ok ? "yes" : "NO  <-- fix path");
    }
    fprintf(stderr, "%d/%d dataset paths readable\n", found, BENCH_NUM_DATASETS);
    return found;
}

#ifdef BENCH_DATASETS_ENABLE_HDF5
static inline hid_t bench_dataset_h5type(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? H5T_IEEE_F64LE : H5T_IEEE_F32LE;  /* file type */
}

static inline hid_t bench_dataset_h5native(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
}

static inline void bench_dataset_h5dims(const bench_dataset_t *d, hsize_t out[]) {
    for (int i = 0; i < d->rank; ++i) out[i] = (hsize_t)d->dims[i];
}
#endif /* BENCH_DATASETS_ENABLE_HDF5 */

#ifdef BENCH_DATASETS_ENABLE_PRESSIO
static inline enum pressio_dtype bench_dataset_pressio_dtype(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? pressio_double_dtype : pressio_float_dtype;
}

static inline void bench_dataset_pressio_dims(const bench_dataset_t *d, size_t out[]) {
    for (int i = 0; i < d->rank; ++i) out[i] = d->dims[d->rank - 1 - i];
}
#endif /* BENCH_DATASETS_ENABLE_PRESSIO */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BENCH_DATASETS_H */