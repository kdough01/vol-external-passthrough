#ifndef BENCH_CONFIG_H
#define BENCH_CONFIG_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BENCH_DATA_ROOT
#define BENCH_DATA_ROOT "/lcrc/project/ECP-EZ/public/compression"
#endif

#define BENCH_MAX_RANK 4

typedef enum {
    BENCH_F32 = 0,   /* little-endian (.f32)      */
    BENCH_F64 = 1    /* little-endian (.f64/.d64) */
} bench_dtype_t;

typedef enum {
    BENCH_BOUND_ABS  = 0,  /* pointwise absolute error bound   */
    BENCH_BOUND_REL  = 1,  /* value-range relative error bound  */
    BENCH_BOUND_NONE = 2   /* rate/precision mode: NO a priori error bound.
                            * Lossy, but the configured knob is bits/value, not
                            * an error tolerance. Fidelity is reported (RMSE,
                            * maxae) and never gated. */
} bench_bound_mode_t;

typedef enum {
    BENCH_SRC_RAW  = 0,
    BENCH_SRC_HDF5 = 1
} bench_src_t;

typedef enum {
    BENCH_XFORM_NONE  = 0,
    BENCH_XFORM_LOG1P = 1,   /* y = log1p(x); non-negative fields only        */
    BENCH_XFORM_ASINH = 2    /* y = asinh(x); signed-safe "log" (velocity)    */
} bench_xform_t;

typedef struct {
    const char        *name;        /* short id used in output filenames/logs */
    const char        *path;        /* raw field, or the .h5/.nc4 container    */
    int                rank;        /* number of dimensions (RAW: required)    */
    size_t             dims[BENCH_MAX_RANK]; /* row-major (HDF5 order)         */
    bench_dtype_t      dtype;
    bench_bound_mode_t bound_mode;  /* NOMINAL bound mode for this field      */
    double             bound;       /* NOMINAL bound value                    */
    double             assumed_range; /* fallback range when none measured     */
    int                gpu_suitable;/* 1 = large/contiguous enough for GPU codec */
    const char        *note;        /* provenance / caveats                   */
    bench_src_t        src;         /* BENCH_SRC_RAW (default/0) or _HDF5      */
    const char        *h5dset;      /* HDF5 src: dataset path inside file      */
    bench_xform_t      xform;
} bench_dataset_t;

static const bench_dataset_t BENCH_DATASETS[] = {

    {
        "miranda",
        BENCH_DATA_ROOT "/Miranda/SDRBENCH-Miranda-256x384x384/density.d64",
        3, {256, 384, 384, 0}, BENCH_F64,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "Clean d64; cuszp returns RMSE~2.7e-4 here. SDRBench names it density.d64.",
        BENCH_SRC_RAW, NULL
    },

    {
        "ocean_temp",
        BENCH_DATA_ROOT "/oceanbox/Tobbeholmane_0001.nc",
        0, {0, 0, 0, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "NetCDF-4 (HDF5-backed). Confirm via `ncdump -k`. Shape/type read at load. "
        "scale_factor/add_offset packing NOT applied.",
        BENCH_SRC_HDF5, "/temp"
    },

    {
        "einspline37",
        BENCH_DATA_ROOT "/QMCPACK-bigdata/einspline.tile_37-1-242-23-8.spin_0.tw_0.l0u6144.g112x66x66.dat",
        1, {13560851520 / 4, 0, 0, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "Raw headerless float32 dump; 1D flat for the codec. 12.6 GiB -> needs "
        "~26 GiB RAM for hbuf+rbuf. Check your PBS mem= request.",
        BENCH_SRC_RAW, NULL
    },

    {
        "scale-T",
        BENCH_DATA_ROOT "/scale-letkf/T-98x1200x1200.f32",
        3, {98, 1200, 1200, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "SCALE-LETKF air temperature. Smooth, well-correlated; representative easy case.",
        BENCH_SRC_RAW, NULL, BENCH_XFORM_LOG1P
    },

    {
        "s3d",
        BENCH_DATA_ROOT "/S3D/stat_planar.1.1000E-03.field.mpi",
        3, {500, 500, 500, 0}, BENCH_F64,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "SDRBench lists S3D as f64 (.d64) despite one stray f32 line on the site.",
        BENCH_SRC_RAW, NULL
    },

    {
        "nyx_baryon",
        BENCH_DATA_ROOT "/NYX-Zarija/z42_n512_l10.h5",
        0, {0, 0, 0, 0}, BENCH_F32,
        BENCH_BOUND_REL, 1e-3, 0.0, 1,
        "NYX 512^3.",
        BENCH_SRC_HDF5, "/native_fields/baryon_density", BENCH_XFORM_LOG1P
    },
};

#define BENCH_NUM_DATASETS \
    ((int)(sizeof(BENCH_DATASETS) / sizeof(BENCH_DATASETS[0])))

static inline size_t bench_dtype_size(bench_dtype_t t) {
    return (t == BENCH_F64) ? 8u : 4u;
}
static inline const char *bench_dtype_name(bench_dtype_t t) {
    return (t == BENCH_F64) ? "double" : "float";
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
    switch (m) {
        case BENCH_BOUND_ABS:  return "abs";
        case BENCH_BOUND_REL:  return "rel";
        case BENCH_BOUND_NONE: return "none";
        default:               return "?";
    }
}
static inline const char *bench_src_name(bench_src_t s) {
    return (s == BENCH_SRC_HDF5) ? "hdf5" : "raw";
}
static inline double bench_gib(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}
static inline const bench_dataset_t *bench_dataset_by_name(const char *name) {
    for (int i = 0; i < BENCH_NUM_DATASETS; ++i)
        if (name && BENCH_DATASETS[i].name &&
            0 == strcmp(name, BENCH_DATASETS[i].name))
            return &BENCH_DATASETS[i];
    return NULL;
}

/* MemAvailable from /proc/meminfo, in bytes. 0 if unknown.
 * Used to skip a dataset cleanly instead of getting OOM-killed halfway
 * through a sweep (einspline37 needs ~26 GiB for hbuf+rbuf alone). */
static inline size_t bench_mem_available_bytes(void) {
    FILE  *fp = fopen("/proc/meminfo", "r");
    char   line[256];
    size_t kb = 0;
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp)) {
        if (0 == strncmp(line, "MemAvailable:", 13)) {
            if (1 == sscanf(line + 13, "%zu", &kb)) break;
            kb = 0;
        }
    }
    fclose(fp);
    return kb * 1024u;
}

static inline int bench_datasets_validate(void) {
    int    found = 0;
    size_t avail = bench_mem_available_bytes();

    fprintf(stderr, "%-14s %-6s %-4s %-6s %-22s %-10s %-10s %s\n",
            "name", "dtype", "rank", "src", "dims", "MiB", "needGiB", "exists");
    for (int i = 0; i < BENCH_NUM_DATASETS; ++i) {
        const bench_dataset_t *d = &BENCH_DATASETS[i];
        char dims[64]; size_t off = 0;
        size_t nb = 0;
        if (d->src == BENCH_SRC_HDF5 && d->rank == 0) {
            snprintf(dims, sizeof(dims), "(from file)");
        } else {
            for (int k = 0; k < d->rank; ++k)
                off += (size_t)snprintf(dims + off, sizeof(dims) - off,
                                        k ? "x%zu" : "%zu", d->dims[k]);
            nb = bench_num_bytes(d);
        }
        int ok = (access(d->path, R_OK) == 0);
        found += ok;
        fprintf(stderr, "%-14s %-6s %-4d %-6s %-22s %-10.1f %-10.1f %s\n",
                d->name, bench_dtype_name(d->dtype), d->rank,
                bench_src_name(d->src), dims,
                nb / (1024.0 * 1024.0), bench_gib(2 * nb),
                ok ? "yes" : "NO  <-- fix path");
    }
    if (avail)
        fprintf(stderr, "MemAvailable: %.1f GiB "
                        "(needGiB is hbuf+rbuf only; the connector needs more)\n",
                bench_gib(avail));
    fprintf(stderr, "%d/%d dataset paths readable\n", found, BENCH_NUM_DATASETS);
    return found;
}

static inline const char *bench_xform_name(bench_xform_t x) {
    switch (x) {
        case BENCH_XFORM_LOG1P: return "log1p";
        case BENCH_XFORM_ASINH: return "asinh";
        default:                return "none";
    }
}
static inline void bench_apply_xform(void *buf, size_t nelem,
                                     bench_dtype_t dt, bench_xform_t xf) {
    if (xf == BENCH_XFORM_NONE) return;
    for (size_t i = 0; i < nelem; ++i) {
        if (dt == BENCH_F64) {
            double v = ((double *)buf)[i];
            ((double *)buf)[i] = (xf == BENCH_XFORM_LOG1P) ? log1p(v) : asinh(v);
        } else {
            float v = ((float *)buf)[i];
            ((float *)buf)[i] = (xf == BENCH_XFORM_LOG1P) ? log1pf(v) : asinhf(v);
        }
    }
}

static inline void *bench_load_raw(const bench_dataset_t *d, size_t *out_bytes) {
    size_t want = bench_num_bytes(d);
    FILE *fp = fopen(d->path, "rb");
    if (!fp) {
        fprintf(stderr, "bench_load_raw: cannot open %s\n", d->path);
        return NULL;
    }
    void *buf = malloc(want);
    if (!buf) {
        fprintf(stderr, "bench_load_raw: OOM (%zu bytes) for %s\n", want, d->name);
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, want, fp);
    fclose(fp);
    if (got != want) {
        fprintf(stderr, "bench_load_raw: short read %s (%zu/%zu bytes)\n",
                d->path, got, want);
        free(buf);
        return NULL;
    }
    if (out_bytes) *out_bytes = want;
    return buf;
}

#ifdef BENCH_CONFIG_ENABLE_HDF5
static inline void *bench_load_h5(const bench_dataset_t *in,
                                  bench_dataset_t *resolved,
                                  size_t *out_bytes) {
    if (!in->h5dset || !in->h5dset[0]) {
        fprintf(stderr, "bench_load_h5: '%s' has BENCH_SRC_HDF5 but h5dset is unset.\n",
                in->name);
        return NULL;
    }
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_vol(fapl, H5VL_NATIVE, NULL);   /* read the source natively; bypass passthrough */
    hid_t fid = H5Fopen(in->path, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (fid < 0) {
        fprintf(stderr, "bench_load_h5: H5Fopen failed: %s "
                "(not HDF5? NetCDF-3/classic is unreadable here)\n", in->path);
        return NULL;
    }
    hid_t did = H5Dopen2(fid, in->h5dset, H5P_DEFAULT);
    if (did < 0) {
        fprintf(stderr, "bench_load_h5: dataset '%s' not found in %s "
                "(check `h5ls -r`)\n", in->h5dset, in->path);
        H5Fclose(fid);
        return NULL;
    }

    hid_t sid  = H5Dget_space(did);
    int   rank = H5Sget_simple_extent_ndims(sid);
    if (rank < 1 || rank > BENCH_MAX_RANK) {
        fprintf(stderr, "bench_load_h5: rank %d unsupported (1..%d) for %s\n",
                rank, BENCH_MAX_RANK, in->h5dset);
        H5Sclose(sid); H5Dclose(did); H5Fclose(fid);
        return NULL;
    }
    hsize_t hdims[BENCH_MAX_RANK];
    H5Sget_simple_extent_dims(sid, hdims, NULL);

    hid_t  ftype  = H5Dget_type(did);
    int    tclass = (int)H5Tget_class(ftype);
    size_t tsize  = H5Tget_size(ftype);
    bench_dtype_t dt;
    if (tclass == (int)H5T_FLOAT && tsize == 4)      dt = BENCH_F32;
    else if (tclass == (int)H5T_FLOAT && tsize == 8) dt = BENCH_F64;
    else {
        fprintf(stderr, "bench_load_h5: unsupported type (class=%d size=%zu) for %s; "
                "only float32/float64 handled.\n", tclass, tsize, in->h5dset);
        H5Tclose(ftype); H5Sclose(sid); H5Dclose(did); H5Fclose(fid);
        return NULL;
    }

    *resolved       = *in;
    resolved->rank  = rank;
    resolved->dtype = dt;
    resolved->src   = BENCH_SRC_HDF5;
    for (int i = 0; i < rank; ++i)              resolved->dims[i] = (size_t)hdims[i];
    for (int i = rank; i < BENCH_MAX_RANK; ++i) resolved->dims[i] = 0;

    size_t nbytes = bench_num_bytes(resolved);
    void  *buf    = malloc(nbytes);
    if (!buf) {
        fprintf(stderr, "bench_load_h5: OOM (%zu bytes) for %s\n", nbytes, in->name);
        H5Tclose(ftype); H5Sclose(sid); H5Dclose(did); H5Fclose(fid);
        return NULL;
    }
    hid_t  mtype = (dt == BENCH_F64) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    herr_t rc    = H5Dread(did, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

    H5Tclose(ftype); H5Sclose(sid); H5Dclose(did); H5Fclose(fid);
    if (rc < 0) {
        fprintf(stderr, "bench_load_h5: H5Dread failed for %s\n", in->h5dset);
        free(buf);
        return NULL;
    }
    if (out_bytes) *out_bytes = nbytes;
    return buf;
}
#endif /* BENCH_CONFIG_ENABLE_HDF5 */

static inline void *bench_load_field(const bench_dataset_t *in,
                                     bench_dataset_t *resolved,
                                     size_t *out_bytes) {
    if (in->src == BENCH_SRC_HDF5) {
#ifdef BENCH_CONFIG_ENABLE_HDF5
        return bench_load_h5(in, resolved, out_bytes);
#else
        fprintf(stderr, "bench_load_field: '%s' is an HDF5 source but this TU was "
                "built without -DBENCH_CONFIG_ENABLE_HDF5.\n", in->name);
        return NULL;
#endif
    }
    *resolved = *in;                 /* raw: descriptor is already complete */
    return bench_load_raw(in, out_bytes);
}

#ifdef BENCH_CONFIG_ENABLE_HDF5
static inline hid_t bench_dataset_h5type(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? H5T_IEEE_F64LE : H5T_IEEE_F32LE;  /* file type */
}
static inline hid_t bench_dataset_h5native(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
}
static inline void bench_dataset_h5dims(const bench_dataset_t *d, hsize_t out[]) {
    for (int i = 0; i < d->rank; ++i) out[i] = (hsize_t)d->dims[i];
}
#endif /* BENCH_CONFIG_ENABLE_HDF5 */

#ifdef BENCH_CONFIG_ENABLE_PRESSIO
static inline enum pressio_dtype bench_dataset_pressio_dtype(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? pressio_double_dtype : pressio_float_dtype;
}
static inline void bench_dataset_pressio_dims(const bench_dataset_t *d, size_t out[]) {
    /* pressio wants fastest-varying dimension first (reverse of HDF5 order) */
    for (int i = 0; i < d->rank; ++i) out[i] = d->dims[d->rank - 1 - i];
}
#endif /* BENCH_CONFIG_ENABLE_PRESSIO */

typedef enum {
    BENCH_CPU_CODEC = 0,
    BENCH_GPU_CODEC = 1
} bench_codec_kind_t;

typedef struct {
    const char        *name;          /* config label / dataset suffix / CSV tag  */
    const char        *pressio_id;    /* compressor id for make_dcpl; NULL=default*/
    const char        *opts_json;     /* literal libpressio options JSON          */
    bench_codec_kind_t kind;
    int                lossless;      /* 1 => bit-exact expected                  */

    bench_bound_mode_t cfg_bound_mode;
    double             cfg_bound;

    const char        *stream_opt_key;/* GPU: userptr key for the cudaStream_t    */
    const char        *note;
} bench_compressor_t;

static const bench_compressor_t BENCH_COMPRESSORS[] = {
    { "noop", "noop", NULL, BENCH_CPU_CODEC, 1,
      BENCH_BOUND_ABS, 0.0, NULL,
      "Connector default (no compressor). Bit-exact baseline; isolates overhead." },

    { "cuszp_1e3", "cuszp",
      "{\"pressio:abs\":1e-3,\"cuszp:mode_str\":\"outlier\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, "cuszp:cuda_stream",
      "GPU abs 1e-3, outlier mode." },

    { "cuszp_1e6", "cuszp",
      "{\"pressio:abs\":1e-6,\"cuszp:mode_str\":\"outlier\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, "cuszp:cuda_stream",
      "GPU abs 1e-6, outlier mode." },

    { "sz3_1e3", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "CPU error-bounded lossy, abs 1e-3." },

    { "sz3_1e6", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "CPU error-bounded lossy, abs 1e-6 (tighter)." },

    { "bzip2", "bzip2",
      "{\"bzip2:block_size\":9}",
      BENCH_CPU_CODEC, 1, BENCH_BOUND_ABS, 0.0, NULL,
      "CPU lossless, general purpose. CPU comparator." },

    /* --- ZFP fixed-rate, matched CPU/GPU pairs ---------------------------- */

    { "zfp_cpu_r4", "zfp",
      "{\"zfp:rate\":4.0,\"zfp:wra\":0,"
       "\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP serial fixed-rate 4 bits/value. Pairs with zfp_gpu_r4." },
    { "zfp_gpu_r4", "zfp",
      "{\"zfp:rate\":4.0,\"zfp:wra\":0,"
       "\"zfp:execution\":2,\"zfp:execution_name\":\"cuda\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP CUDA fixed-rate 4 bits/value. Pairs with zfp_cpu_r4." },

    { "zfp_cpu_r8", "zfp",
      "{\"zfp:rate\":8.0,\"zfp:wra\":0,"
       "\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP serial fixed-rate 8 bits/value. Pairs with zfp_gpu_r8." },
    { "zfp_gpu_r8", "zfp",
      "{\"zfp:rate\":8.0,\"zfp:wra\":0,"
       "\"zfp:execution\":2,\"zfp:execution_name\":\"cuda\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP CUDA fixed-rate 8 bits/value. Pairs with zfp_cpu_r8." },

    { "zfp_cpu_r12", "zfp",
      "{\"zfp:rate\":12.0,\"zfp:wra\":0,"
       "\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP serial fixed-rate 12 bits/value. Pairs with zfp_gpu_r12." },
    { "zfp_gpu_r12", "zfp",
      "{\"zfp:rate\":12.0,\"zfp:wra\":0,"
       "\"zfp:execution\":2,\"zfp:execution_name\":\"cuda\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP CUDA fixed-rate 12 bits/value. Pairs with zfp_cpu_r12." },

    { "zfp_cpu_r16", "zfp",
      "{\"zfp:rate\":16.0,\"zfp:wra\":0,"
       "\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP serial fixed-rate 16 bits/value. Pairs with zfp_gpu_r16." },
    { "zfp_gpu_r16", "zfp",
      "{\"zfp:rate\":16.0,\"zfp:wra\":0,"
       "\"zfp:execution\":2,\"zfp:execution_name\":\"cuda\"}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "ZFP CUDA fixed-rate 16 bits/value. Pairs with zfp_cpu_r16." },

    /* CPU-only accuracy arms. NOT comparable to the GPU rows above -- zfp has
     * no CUDA accuracy mode. Keep them for the mode comparison, plot them
     * separately from the rate pairs. */
    { "zfp_1e3", "zfp",
      "{\"zfp:accuracy\":1e-3,\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "ZFP serial fixed-accuracy 1e-3 (CPU only; no CUDA equivalent)." },
    { "zfp_1e6", "zfp",
      "{\"zfp:accuracy\":1e-6,\"zfp:execution\":0,\"zfp:execution_name\":\"serial\"}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "ZFP serial fixed-accuracy 1e-6 (CPU only; no CUDA equivalent)." },

      { "szx_1e3", "szx",
      "{\"pressio:abs\":1e-3}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "SZx ultra-fast error-bounded, abs 1e-3. Matches sz3_1e3 / cuszp_1e3." },

    { "szx_1e6", "szx",
      "{\"pressio:abs\":1e-6}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "SZx ultra-fast error-bounded, abs 1e-6. Matches sz3_1e6 / cuszp_1e6." },

    /* --- JSON-path chunking (N chunks via opts_json; do NOT set VOL_COMP_* env).
     *     chunk_n=8 divides every static dataset. noop+pressio is omitted:
     *     rejected by the VOL by design. --- */

    { "noop_vjson", "noop",
      "{\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 1, BENCH_BOUND_ABS, 0.0, NULL,
      "noop, VOL chunking via opts_json." },

    { "bzip2_vjson", "bzip2",
      "{\"bzip2:block_size\":9,\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 1, BENCH_BOUND_ABS, 0.0, NULL,
      "bzip2, VOL chunking via opts_json." },
    { "bzip2_pjson", "bzip2",
      "{\"bzip2:block_size\":9,\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 1, BENCH_BOUND_ABS, 0.0, NULL,
      "bzip2, pressio chunking via opts_json." },

    { "sz3_1e3_vjson", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3,"
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "sz3 1e-3, VOL chunking via opts_json." },
    { "sz3_1e3_pjson", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3,"
       "\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "sz3 1e-3, pressio chunking via opts_json." },

    { "sz3_1e6_vjson", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6,"
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "sz3 1e-6, VOL chunking via opts_json." },
    { "sz3_1e6_pjson", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6,"
       "\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "sz3 1e-6, pressio chunking via opts_json." },

    { "szx_1e3_vjson", "szx",
      "{\"pressio:abs\":1e-3,\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "szx 1e-3, VOL chunking via opts_json." },
    { "szx_1e3_pjson", "szx",
      "{\"pressio:abs\":1e-3,\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, NULL,
      "szx 1e-3, pressio chunking via opts_json." },

    { "szx_1e6_vjson", "szx",
      "{\"pressio:abs\":1e-6,\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "szx 1e-6, VOL chunking via opts_json." },
    { "szx_1e6_pjson", "szx",
      "{\"pressio:abs\":1e-6,\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, NULL,
      "szx 1e-6, pressio chunking via opts_json." },

    { "cuszp_1e3_vjson", "cuszp",
      "{\"pressio:abs\":1e-3,\"cuszp:mode_str\":\"outlier\","
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, "cuszp:cuda_stream",
      "cuszp 1e-3, VOL chunking via opts_json." },
    { "cuszp_1e3_pjson", "cuszp",
      "{\"pressio:abs\":1e-3,\"cuszp:mode_str\":\"outlier\","
       "\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-3, "cuszp:cuda_stream",
      "cuszp 1e-3, pressio chunking via opts_json." },

    { "cuszp_1e6_vjson", "cuszp",
      "{\"pressio:abs\":1e-6,\"cuszp:mode_str\":\"outlier\","
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, "cuszp:cuda_stream",
      "cuszp 1e-6, VOL chunking via opts_json." },
    { "cuszp_1e6_pjson", "cuszp",
      "{\"pressio:abs\":1e-6,\"cuszp:mode_str\":\"outlier\","
       "\"vol:chunking_mode\":\"pressio\",\"vol:chunk_n\":8}",
      BENCH_GPU_CODEC, 0, BENCH_BOUND_ABS, 1e-6, "cuszp:cuda_stream",
      "cuszp 1e-6, pressio chunking via opts_json." },

    { "sperr_pwe1e3", "sperr",
      "{\"sperr:mode_str\":\"pwe\",\"sperr:tolerance\":1e-3}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "SPERR point-wise error 1e-3, native container (one whole-volume "
      "stream). The uniform-percentage arm." },

    { "sperr_pwe1e6", "sperr",
      "{\"sperr:mode_str\":\"pwe\",\"sperr:tolerance\":1e-6}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "SPERR point-wise error 1e-6. Tighter bound => larger stream => more "
      "absolute bytes saved at a given percentage." },

    { "sperr_pwe1e3_v8", "sperr",
      "{\"sperr:mode_str\":\"pwe\",\"sperr:tolerance\":1e-3,"
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "SPERR 1e-3, VOL chunking N=8 via opts_json. The per-chunk-fidelity arm: "
      "8 independent streams, each truncatable to its own percentage. "
      "chunk_n must divide dims[0] or vol_slab_dims rejects the write." },

    { "sperr_bpp2_v8", "sperr",
      "{\"sperr:mode_str\":\"bpp\",\"sperr:tolerance\":2.0,"
       "\"vol:chunking_mode\":\"vol\",\"vol:chunk_n\":8}",
      BENCH_CPU_CODEC, 0, BENCH_BOUND_NONE, 0.0, NULL,
      "SPERR fixed-rate 2 bits/value, VOL chunking N=8. Fixed rate makes every "
      "chunk's stream a predictable size, so bytes-read scales cleanly with "
      "percentage -- a cleaner mechanism plot than a bounded mode gives." },
};

#define BENCH_NUM_COMPRESSORS \
    ((int)(sizeof(BENCH_COMPRESSORS) / sizeof(BENCH_COMPRESSORS[0])))

static inline const char *bench_codec_kind_name(bench_codec_kind_t k) {
    return (k == BENCH_GPU_CODEC) ? "gpu" : "cpu";
}
static inline const bench_compressor_t *bench_compressor_by_name(const char *name) {
    for (int i = 0; i < BENCH_NUM_COMPRESSORS; ++i)
        if (name && 0 == strcmp(name, BENCH_COMPRESSORS[i].name))
            return &BENCH_COMPRESSORS[i];
    return NULL;
}
/* The options JSON to hand to make_dcpl / pressio: literal opts_json or "{}". */
static inline const char *bench_compressor_opts(const bench_compressor_t *c) {
    return (c->opts_json && c->opts_json[0]) ? c->opts_json : "{}";
}

/* zfp_type from zfp.h: none=0, int32=1, int64=2, float=3, double=4. */
static inline int bench_zfp_type_code(const bench_dataset_t *d) {
    return (d->dtype == BENCH_F64) ? 4 : 3;
}

static inline double bench_zfp_rate_from_json(const bench_compressor_t *c) {
    const char *p;
    if (!c || !c->opts_json) return 0.0;
    p = strstr(c->opts_json, "\"zfp:rate\"");
    if (!p) return 0.0;
    p += 10;                                    /* strlen("\"zfp:rate\"") */
    while (*p == ' ' || *p == '\t') p++;
    if (*p != ':') return 0.0;
    return atof(p + 1);
}

static inline double bench_abs_threshold(const bench_compressor_t *c,
                                         const bench_dataset_t *d,
                                         double measured_range) {
    double range;
    if (!c) return 0.0;
    if (c->lossless) return 0.0;                      /* bit-exact expected */
    if (c->cfg_bound_mode == BENCH_BOUND_NONE) return 0.0;  /* no bound to check */
    if (c->cfg_bound_mode == BENCH_BOUND_ABS) return c->cfg_bound;

    range = (measured_range > 0.0) ? measured_range
          : (d && d->assumed_range > 0.0) ? d->assumed_range
          : 1.0;
    return c->cfg_bound * range;
}

/* 1 if a pass/fail fidelity verdict is meaningful for this entry. Rate-mode
 * entries are lossy with no bound, so their maxae must be reported, not judged. */
static inline int bench_bound_is_checkable(const bench_compressor_t *c) {
    if (!c) return 0;
    if (c->lossless) return 1;
    return c->cfg_bound_mode != BENCH_BOUND_NONE;
}

/* Number of chunks declared in opts_json ("vol:chunk_n"). 1 if absent. */
static inline int bench_compressor_chunk_n(const bench_compressor_t *c) {
    const char *p;
    int n;
    if (!c || !c->opts_json) return 1;
    p = strstr(c->opts_json, "\"vol:chunk_n\"");
    if (!p) return 1;
    p = strchr(p, ':');
    if (!p) return 1;
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    n = atoi(p);
    return n > 0 ? n : 1;
}

/* Effective chunk count: VOL_COMP_CHUNK_N env overrides opts_json, matching
 * the connector's own precedence in H5VL_pass_through_ext_chunk_bytes(). */
static inline int bench_effective_chunk_n(const bench_compressor_t *c) {
    const char *e = getenv("VOL_COMP_CHUNK_N");
    if (e && *e) {
        int n = atoi(e);
        if (n > 0) return n;
    }
    return bench_compressor_chunk_n(c);
}

static inline int bench_compressor_opts_json(const bench_compressor_t *c,
                                             const bench_dataset_t *d,
                                             char *buf, size_t n) {
    int ret;
    int dbg = (getenv("BENCH_DEBUG") != NULL);

    /* 0) zfp fixed-rate: zfp_stream_set_rate() takes (rate, type, dims, wra),
     *    so type and dims must accompany the rate -- and both are properties of
     *    the dataset, not of the compressor entry. Splice them into the literal
     *    JSON here rather than duplicating a table row per (rate x dtype x rank)
     *    combination. Only entries that actually set zfp:rate are touched, so
     *    the accuracy entries keep their own mode selection. */
    if (c->opts_json && c->opts_json[0] &&
        c->pressio_id && 0 == strcmp(c->pressio_id, "zfp") &&
        strstr(c->opts_json, "\"zfp:rate\"")) {
        size_t len = strlen(c->opts_json);
        if (len >= 2 && c->opts_json[len - 1] == '}') {
            ret = snprintf(buf, n, "%.*s,\"zfp:type\":%d,\"zfp:dims\":%d}",
                           (int)(len - 1), c->opts_json,
                           bench_zfp_type_code(d), d->rank);
            if (dbg) fprintf(stderr, "[dbg opts] %-16s %-12s ZFPRATE  -> %s\n",
                             c->name, d->name, buf);
            return ret;
        }
    }

    /* 1) An explicit literal opts_json on the compressor entry wins outright.
     *    Every lossy entry in the table above takes this path. */
    if (c->opts_json && c->opts_json[0]) {
        ret = snprintf(buf, n, "%s", c->opts_json);
        if (dbg) fprintf(stderr, "[dbg opts] %-16s %-12s LITERAL  -> %s\n",
                         c->name, d->name, buf);
        return ret;
    }

    /* 2) Lossless codecs take no bound. */
    if (c->lossless) {
        ret = snprintf(buf, n, "{}");
        if (dbg) fprintf(stderr, "[dbg opts] %-16s %-12s LOSSLESS -> %s\n",
                         c->name, d->name, buf);
        return ret;
    }

    /* 3) No literal JSON: build one from the entry's declared bound. cuSZp is
     *    natively absolute and its libpressio plugin's only bound knob is
     *    pressio:abs, so a relative request is converted using the dataset's
     *    assumed_range. (The old unreachable `strcmp(c->name, "cuszp")` branch
     *    is gone -- no entry is named exactly "cuszp", so it never ran.) */
    {
        const int is_cuszp = (c->pressio_id && 0 == strcmp(c->pressio_id, "cuszp"));
        double    absb;

        if (c->cfg_bound_mode == BENCH_BOUND_NONE) {
            /* Rate-controlled entry with no literal JSON: nothing derivable. */
            ret = snprintf(buf, n, "{}");
            if (dbg) fprintf(stderr, "[dbg opts] %-16s %-12s NOBOUND  -> %s\n",
                             c->name, d->name, buf);
            return ret;
        }

        if (c->cfg_bound_mode == BENCH_BOUND_ABS) {
            absb = c->cfg_bound;
        } else {
            double range = (d->assumed_range > 0.0) ? d->assumed_range : 1.0;
            absb = c->cfg_bound * range;
        }

        if (is_cuszp)
            ret = snprintf(buf, n,
                "{\"pressio:abs\": %.10e, \"cuszp:mode_str\": \"outlier\"}", absb);
        else if (c->cfg_bound_mode == BENCH_BOUND_ABS)
            ret = snprintf(buf, n, "{\"pressio:abs\": %.10e}", absb);
        else
            ret = snprintf(buf, n, "{\"pressio:rel\": %.10e}", c->cfg_bound);

        if (dbg) fprintf(stderr, "[dbg opts] %-16s %-12s DERIVED  -> %s\n",
                         c->name, d->name, buf);
        return ret;
    }
}

static inline int bench_compressor_vol_info_json(const bench_compressor_t *c,
                                                 const bench_dataset_t *d,
                                                 char *buf, size_t n) {
    char opts[256];
    bench_compressor_opts_json(c, d, opts, sizeof(opts));
    if (!c->pressio_id)
        return snprintf(buf, n, "{\"compressor\": null, \"options\": %s}", opts);
    return snprintf(buf, n,
        "{\"compressor\": \"%s\", \"options\": %s}", c->pressio_id, opts);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BENCH_CONFIG_H */