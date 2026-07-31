#include <hdf5.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <string>

#define BENCH_CONFIG_ENABLE_HDF5
#include "bench_config.h"
#include "bench_timing.h"

#define FID_BZIP2_DEFAULT   307
#define FID_ZFP_DEFAULT     32013
#define FID_SZ3_DEFAULT     32024
#define FID_LP_DEFAULT      32026

#if defined(__has_include)
#  if __has_include(<H5Zzfp_plugin.h>)
#    include <H5Zzfp_plugin.h>
#    define VOL_HAVE_H5ZZFP 1
#  endif
#endif

typedef enum {
    FILTER_DEFLATE, FILTER_BZIP2, FILTER_ZFP, FILTER_SZ3, FILTER_LIBPRESSIO
} filter_backend_t;

static H5Z_filter_t env_fid(const char *var, int dflt) {
    const char *e = getenv(var);
    return (H5Z_filter_t)(e && *e ? atoi(e) : dflt);
}

static H5Z_filter_t backend_fid(filter_backend_t b) {
    switch (b) {
        case FILTER_DEFLATE: return H5Z_FILTER_DEFLATE;
        case FILTER_BZIP2:   return env_fid("H5Z_BZIP2_ID", FID_BZIP2_DEFAULT);
        case FILTER_ZFP:     return env_fid("H5Z_ZFP_ID",   FID_ZFP_DEFAULT);
        case FILTER_SZ3:     return env_fid("H5Z_SZ3_ID",   FID_SZ3_DEFAULT);
        default:             return env_fid("LIBPRESSIO_H5Z_FILTER_ID", FID_LP_DEFAULT);
    }
}

static const char *backend_name(filter_backend_t b) {
    switch (b) {
        case FILTER_DEFLATE: return "deflate";
        case FILTER_BZIP2:   return "bzip2";
        case FILTER_ZFP:     return "zfp";
        case FILTER_SZ3:     return "sz3";
        default:             return "libpressio";
    }
}

/* 1 when this backend runs the same codec as the VOL entry of the same name, so
 * ratio and fidelity comparisons are valid. Deflate is not in that set. */
static int backend_codec_matched(filter_backend_t b) {
    return b != FILTER_DEFLATE;
}

/* Identical to bench_vol_timing.cc's XCSV_HEADER so the two merge directly. */
static const char *XCSV_HEADER =
    "dataset,compressor,codec_kind,chunk_n,rep,"
    "logical_bytes,stored_bytes,ratio,"
    "create_ms,write_ms,flush_ms,close_ms,open_ms,read_ms,"
    "rmse,abs_thresh,maxae,bound_ok\n";

typedef struct { double min, max, mean, rmse, maxae; } bench_stats;

static bench_stats compute_stats(const void *orig, const void *dec,
                                 size_t nelem, bench_dtype_t dt) {
    double mn = DBL_MAX, mx = -DBL_MAX, sum = 0.0, se = 0.0, maxae = 0.0;
    for (size_t i = 0; i < nelem; ++i) {
        double o = (dt == BENCH_F64) ? ((const double *)orig)[i]
                                     : (double)((const float *)orig)[i];
        double v = (dt == BENCH_F64) ? ((const double *)dec)[i]
                                     : (double)((const float *)dec)[i];
        if (o < mn) mn = o;
        if (o > mx) mx = o;
        sum += o;
        double e = o - v;
        se += e * e;
        double ae = std::fabs(e); if (ae > maxae) maxae = ae;
    }
    bench_stats s;
    s.min = mn; s.max = mx;
    s.mean  = sum / (double)nelem;
    s.rmse  = std::sqrt(se / (double)nelem);
    s.maxae = maxae;
    return s;
}

/* chunk_n splits the SLOWEST-varying dimension, the closest HDF5 analogue to the
 * VOL's flat N-way split. HDF5 tolerates a ragged final chunk but PADS it on
 * disk -- a space cost the VOL container does not pay, and worth mentioning when
 * comparing file sizes. */
static hid_t make_dcpl(const bench_dataset_t *d, filter_backend_t backend,
                       const bench_compressor_t *c, int chunk_n,
                       int deflate_level, double abs_bound) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, chunk);
    if (chunk_n > 1) {
        hsize_t per = (chunk[0] + (hsize_t)chunk_n - 1) / (hsize_t)chunk_n;
        chunk[0] = per < 1 ? 1 : per;
    }
    H5Pset_chunk(dcpl, d->rank, chunk);

    /* Match the VOL: no fill-value writes, allocate late. Otherwise the filter
     * path pays a fill pass the VOL container does not. */
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_LATE);

    switch (backend) {
    case FILTER_DEFLATE:
        H5Pset_deflate(dcpl, (unsigned)deflate_level);
        break;

    case FILTER_BZIP2: {
        /* H5Z-bzip2 takes exactly one cd_value: block size 1..9. That is the
         * same knob as "bzip2:block_size" in bench_config.h, so this is a true
         * same-codec same-parameter comparison against the VOL. */
        unsigned cd[1] = { 9 };
        if (c && c->opts_json) {
            const char *p = strstr(c->opts_json, "\"bzip2:block_size\"");
            if (p && (p = strchr(p, ':'))) cd[0] = (unsigned)atoi(p + 1);
        }
        if (cd[0] < 1 || cd[0] > 9) cd[0] = 9;
        H5Pset_filter(dcpl, backend_fid(backend), H5Z_FLAG_MANDATORY, 1, cd);
        break;
    }

    case FILTER_SZ3: {
        /* CONFIRM against your H5Z-SZ3 README: this assumes
         * [mode, hi32(bound), lo32(bound)] with mode 0 == absolute. The layout
         * differs between H5Z-SZ and H5Z-SZ3. */
        union { double d; unsigned u[2]; } b;
        b.d = abs_bound;
        unsigned cd[3] = { 0u, b.u[0], b.u[1] };
        H5Pset_filter(dcpl, backend_fid(backend), H5Z_FLAG_MANDATORY, 3, cd);
        break;
    }

    case FILTER_ZFP: {
#ifdef VOL_HAVE_H5ZZFP
        size_t       cd_nelmts = 10;
        unsigned int cd_values[10] = {0};
        H5Pset_zfp_accuracy_cdata(abs_bound, cd_nelmts, cd_values);
        H5Pset_filter(dcpl, backend_fid(backend), H5Z_FLAG_MANDATORY,
                      (unsigned)cd_nelmts, cd_values);
#else
        unsigned int cd[1] = {0};
        H5Pset_filter(dcpl, backend_fid(backend), H5Z_FLAG_MANDATORY, 0, cd);
        std::fprintf(stderr,
            "WARNING: H5Zzfp_plugin.h not found -- zfp runs at DEFAULT settings, "
            "NOT accuracy=%g. Add $(spack location -i h5z-zfp)/include to the "
            "include path in tests/CMakeLists.txt.\n", abs_bound);
#endif
        break;
    }
    }

    default: {
        unsigned cd[1] = { 0 };
        H5Pset_filter(dcpl, backend_fid(backend), H5Z_FLAG_MANDATORY, 0, cd);
        std::fprintf(stderr, "WARNING: libpressio filter codec/options not "
                             "wired up -- rows are NOT comparable to the VOL\n");
        break;
    }
    }
    return dcpl;
}

static int run_one(const bench_dataset_t *din, filter_backend_t backend,
                   const bench_compressor_t *c, const char *h5path,
                   int chunk_n, int deflate_level, FILE *csv, FILE *xcsv) {
    bench_dataset_t d;
    size_t raw = 0;
    void *hbuf = bench_load_field(din, &d, &raw);   /* resolves HDF5-sourced dims */
    if (!hbuf) {
        std::fprintf(stderr, "[skip] %s: load failed\n", din->name);
        std::fprintf(xcsv, "%s,%s,cpu,%d,-1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0\n",
                     din->name, (backend == FILTER_DEFLATE) ? "deflate" : c->name,
                     chunk_n);
        std::fflush(xcsv);
        return -1;
    }

    if (d.xform != BENCH_XFORM_NONE)     /* same preprocessing as the VOL harness */
        bench_apply_xform(hbuf, bench_num_elements(&d), d.dtype, d.xform);

    double range = 0.0;
    {
        size_t ne = bench_num_elements(&d);
        double mn = DBL_MAX, mx = -DBL_MAX;
        for (size_t i = 0; i < ne; ++i) {
            double v = (d.dtype == BENCH_F64) ? ((const double *)hbuf)[i]
                                              : (double)((const float *)hbuf)[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        range = mx - mn;
    }

    void *rbuf = std::malloc(raw);
    if (!rbuf) {
        std::fprintf(stderr, "[skip] %s: OOM rbuf\n", d.name);
        std::free(hbuf);
        return -1;
    }

    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(&d, dims);
    hid_t ntype = bench_dataset_h5native(&d);
    hid_t ftype = bench_dataset_h5type(&d);

    const char *cname = (backend == FILTER_DEFLATE) ? "deflate" : c->name;
    const char *kind  = (backend == FILTER_DEFLATE) ? "cpu"
                                                    : bench_codec_kind_name(c->kind);
    const int lossless = (backend == FILTER_DEFLATE || backend == FILTER_BZIP2)
                             ? 1 : c->lossless;
    const double thr = (backend == FILTER_DEFLATE || !backend_codec_matched(backend))
                           ? 0.0 : bench_abs_threshold(c, &d, range);
    const double abs_bound = (backend == FILTER_SZ3 || backend == FILTER_ZFP)
                                 ? bench_abs_threshold(c, &d, range) : 0.0;

    double create_ms = 0, write_ms = 0, flush_ms = 0, close_ms = 0;
    double open_ms = 0, read_ms = 0;
    unsigned long long stored = 0, filesize = 0;
    int rc = 0;

    /* ---------------- WRITE ---------------- */
    {
        BenchCpuTimer ct; ct.start();
        hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        create_ms = ct.stop_ms();
        if (file < 0) { std::fprintf(stderr, "[ERR] H5Fcreate %s\n", h5path); rc = -1; goto cleanup; }

        hid_t space = H5Screate_simple(d.rank, dims, NULL);
        hid_t dcpl  = make_dcpl(&d, backend, c, chunk_n, deflate_level, abs_bound);
        hid_t dset  = H5Dcreate2(file, d.name, ftype, space, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
        if (dset < 0) {
            /* Most common cause: the chunk exceeds HDF5's 4 GiB limit, or the
             * filter plugin is missing. Both are results, not silent skips. */
            std::fprintf(stderr, "[ERR] H5Dcreate2 %s N=%d -- chunk >4GiB, or "
                                 "filter %d unavailable?\n",
                         d.name, chunk_n, (int)backend_fid(backend));
            H5Pclose(dcpl); H5Sclose(space); H5Fclose(file); rc = -1; goto cleanup;
        }

        BenchCpuTimer wt; wt.start();
        herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
        write_ms = wt.stop_ms();

        BenchCpuTimer ft; ft.start();
        (void)H5Fflush(file, H5F_SCOPE_LOCAL);
        flush_ms = ft.stop_ms();

        stored = (unsigned long long)H5Dget_storage_size(dset);  /* == VOL metric */
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(space);

        BenchCpuTimer clt; clt.start();
        H5Fclose(file);
        close_ms = clt.stop_ms();

        if (wret < 0) { std::fprintf(stderr, "[ERR] H5Dwrite %s\n", d.name); rc = -1; goto cleanup; }
    }

    /* ---------------- READ ---------------- */
    {
        std::memset(rbuf, 0, raw);

        BenchCpuTimer ot; ot.start();
        hid_t file = H5Fopen(h5path, H5F_ACC_RDONLY, H5P_DEFAULT);
        open_ms = ot.stop_ms();
        if (file < 0) { std::fprintf(stderr, "[ERR] H5Fopen %s\n", h5path); rc = -1; goto cleanup; }

        H5Fget_filesize(file, (hsize_t *)&filesize);

        hid_t dset = H5Dopen2(file, d.name, H5P_DEFAULT);
        if (dset < 0) { std::fprintf(stderr, "[ERR] H5Dopen2 %s\n", d.name); H5Fclose(file); rc = -1; goto cleanup; }

        BenchCpuTimer rt; rt.start();
        herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        read_ms = rt.stop_ms();

        H5Dclose(dset); H5Fclose(file);
        if (rret < 0) { std::fprintf(stderr, "[ERR] H5Dread %s\n", d.name); rc = -1; goto cleanup; }
    }

    /* ---------------- FIDELITY + EMIT ---------------- */
    {
        size_t nelem = bench_num_elements(&d);
        bench_stats st = compute_stats(hbuf, rbuf, nelem, d.dtype);
        double ratio = stored ? (double)raw / (double)stored : 0.0;
        int bound_ok = (thr > 0.0) ? (st.maxae <= 2.0 * thr) : (st.maxae == 0.0);

        if (lossless && st.maxae != 0.0)
            std::fprintf(stderr, "[FAIL] %s/%s declared lossless but maxae=%.6e\n",
                         d.name, cname, st.maxae);
        else if (thr > 0.0 && st.maxae > 2.0 * thr)
            std::fprintf(stderr, "[FAIL] %s/%s maxae=%.6e exceeds 2x bound %.6e\n",
                         d.name, cname, st.maxae, thr);

        bench_csv_row(csv, d.name, cname, "filter", "write", "total", write_ms, ratio, -1.0);
        bench_csv_row(csv, d.name, cname, "filter", "read",  "total", read_ms, -1.0, st.rmse);

        std::fprintf(xcsv,
            "%s,%s,%s,%d,%d,%llu,%llu,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%.6e,%.6e,%.6e,%d\n",
            d.name, cname, kind, chunk_n, 0,
            (unsigned long long)raw, stored, ratio,
            create_ms, write_ms, flush_ms, close_ms, open_ms, read_ms,
            st.rmse, thr, st.maxae, bound_ok);
        std::fflush(xcsv);

        std::printf("  %-14s %-10s N=%-3d C=%6.2f W=%9.2f F=%8.2f X=%7.2f | "
                    "O=%6.2f R=%9.2f ms  ratio=%6.2fx  file=%.1f MiB  "
                    "RMSE=%.3e maxae=%.3e\n",
                    d.name, cname, chunk_n, create_ms, write_ms, flush_ms,
                    close_ms, open_ms, read_ms, ratio,
                    filesize / (1024.0 * 1024.0), st.rmse, st.maxae);
        std::fflush(stdout);
    }

cleanup:
    /* Record failures as a rep=-1 row rather than leaving a header-only CSV. An
     * unsupported configuration IS a result: einspline37 as a single chunk
     * exceeds HDF5's 4 GiB limit, which the VOL container has no equivalent of. */
    if (rc != 0) {
        std::fprintf(xcsv,
            "%s,%s,%s,%d,-1,%llu,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0\n",
            d.name, cname, kind, chunk_n, (unsigned long long)raw);
        std::fflush(xcsv);
        std::fprintf(stderr, "[RECORDED FAILURE] %s/%s N=%d\n", d.name, cname, chunk_n);
    }
    std::free(rbuf);
    std::free(hbuf);
    return rc;
}

int main(int argc, char **argv) {
    const char *h5path   = "bench_filter.h5";
    const char *csvpath  = "results_filter.csv";
    const char *xcsvpath = NULL;
    filter_backend_t backend = FILTER_BZIP2;      /* same-codec default */
    const char *only_cmp = getenv("BENCH_COMP");
    const char *only_ds  = getenv("BENCH_ONLY");
    int chunk_n = 1, deflate_level = 6, nfail = 0;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--deflate"))    backend = FILTER_DEFLATE;
        else if (!std::strcmp(argv[i], "--bzip2"))      backend = FILTER_BZIP2;
        else if (!std::strcmp(argv[i], "--zfp"))        backend = FILTER_ZFP;
        else if (!std::strcmp(argv[i], "--sz3"))        backend = FILTER_SZ3;
        else if (!std::strcmp(argv[i], "--libpressio")) backend = FILTER_LIBPRESSIO;
        else if (!std::strcmp(argv[i], "--comp")    && i + 1 < argc) only_cmp = argv[++i];
        else if (!std::strcmp(argv[i], "--only")    && i + 1 < argc) only_ds  = argv[++i];
        else if (!std::strcmp(argv[i], "--out")     && i + 1 < argc) csvpath  = argv[++i];
        else if (!std::strcmp(argv[i], "--ext")     && i + 1 < argc) xcsvpath = argv[++i];
        else if (!std::strcmp(argv[i], "--h5")      && i + 1 < argc) h5path   = argv[++i];
        else if (!std::strcmp(argv[i], "--chunk-n") && i + 1 < argc) chunk_n  = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--level")   && i + 1 < argc) deflate_level = atoi(argv[++i]);
    }
    if (chunk_n < 1) chunk_n = 1;

    char xdefault[512];
    if (!xcsvpath) {
        std::snprintf(xdefault, sizeof(xdefault), "%s.ext.csv", csvpath);
        xcsvpath = xdefault;
    }

    std::fprintf(stderr, "[filter] backend=%s id=%d chunk_n=%d\n",
                 backend_name(backend), (int)backend_fid(backend), chunk_n);
    if (H5Zfilter_avail(backend_fid(backend)) <= 0)
        std::fprintf(stderr, "WARNING: filter %d (%s) not available. Set "
                             "HDF5_PLUGIN_PATH, and H5Z_%s_ID if the id differs.\n",
                     (int)backend_fid(backend), backend_name(backend),
                     backend == FILTER_BZIP2 ? "BZIP2" :
                     backend == FILTER_ZFP   ? "ZFP"   :
                     backend == FILTER_SZ3   ? "SZ3"   : "LIBPRESSIO");
    if (!backend_codec_matched(backend))
        std::fprintf(stderr, "NOTE: %s is a DIFFERENT codec from the VOL entries. "
                             "Use for the overhead claim only, never ratio or "
                             "fidelity.\n", backend_name(backend));

    if (getenv("BENCH_DEBUG")) bench_datasets_validate();

    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    FILE *xcsv = std::fopen(xcsvpath, "w");
    if (!xcsv) { std::perror("xcsv"); std::fclose(csv); return 1; }
    std::fputs(XCSV_HEADER, xcsv);

    const size_t mem_avail = bench_mem_available_bytes();

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (only_ds && !std::strstr(only_ds, d->name)) continue;
        if (access(d->path, R_OK) != 0) continue;

        if (d->src != BENCH_SRC_HDF5 && mem_avail) {
            size_t need = 2 * bench_num_bytes(d);
            if (need > (size_t)(0.9 * (double)mem_avail)) {
                std::fprintf(stderr, "[SKIP] %s needs %.1f GiB, %.1f GiB available\n",
                             d->name, bench_gib(need), bench_gib(mem_avail));
                continue;
            }
        }

        if (backend == FILTER_DEFLATE) {
            if (run_one(d, backend, NULL, h5path, chunk_n, deflate_level, csv, xcsv) != 0)
                nfail++;
        } else {
            for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
                const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
                if (only_cmp && std::strcmp(only_cmp, c->name)) continue;
                /* Only run entries whose codec this backend actually implements. */
                if (backend == FILTER_BZIP2 && std::strcmp(c->pressio_id, "bzip2")) continue;
                if (backend == FILTER_SZ3   && std::strcmp(c->pressio_id, "sz3"))   continue;
                if (backend == FILTER_ZFP   && std::strcmp(c->pressio_id, "zfp"))   continue;
                if (run_one(d, backend, c, h5path, chunk_n, deflate_level, csv, xcsv) != 0)
                    nfail++;
            }
        }
        std::remove(h5path);
    }

    std::fclose(csv);
    std::fclose(xcsv);
    std::fprintf(stderr, "wrote %s and %s (%d failed configuration(s) recorded)\n",
                 csvpath, xcsvpath, nfail);
    return nfail ? 2 : 0;   /* non-zero so the PBS script notices */
}