/* ============================================================================
 * bench_filter_timing.cc  --  APPROACH 3 of 3: HDF5 H5Z FILTER
 * ----------------------------------------------------------------------------
 * Runs a compressor through HDF5's chunked filter pipeline: same file format and
 * HDF5 API as the VOL, but compression happens per-chunk inside H5Dwrite via an
 * H5Z filter. This is the architectural comparator the paper's "minimal overhead
 * compared to existing implementations" claim rests on.
 *
 * Backends:
 *   --deflate      built-in gzip (H5Pset_deflate). Always available. CPU
 *                  lossless comparator -- a different codec from sz3/cuszp, so
 *                  use it for the overhead claim, not for ratio or fidelity.
 *   --libpressio   the libpressio H5Z filter, so the SAME codec runs here as in
 *                  the VOL. Apples-to-apples. Needs the plugin on
 *                  HDF5_PLUGIN_PATH and the filter id (see LIBPRESSIO_H5Z_*).
 *
 * WHAT CHANGED FROM THE FIRST VERSION
 * -----------------------------------
 * 1. Emits the SAME 18-column extended schema as bench_vol_timing.cc, so the
 *    three-way figure (filters vs libpressio vs VOL) is a concat + groupby
 *    rather than a schema reconciliation.
 * 2. Fidelity is actually measured. The old version read into rbuf and never
 *    compared it, so it could not contribute to the overhead OR the
 *    rate-distortion figure. RMSE/maxae/bound_ok now come out, thresholded with
 *    bench_abs_threshold() like the VOL harness.
 * 3. HDF5-sourced datasets work. The old local load_field() read raw bytes with
 *    bench_num_bytes(), which is 0 for ocean_temp and nyx_baryon (rank 0, dims
 *    resolved from the file). Now uses bench_load_field() + the resolved
 *    descriptor.
 * 4. The log1p/asinh transform is applied. Without it, scale-T and nyx_baryon
 *    were being compressed as raw values here and as transformed values in the
 *    VOL harness -- the ratios were not comparable at all.
 * 5. Chunk-count sweep (--chunk-n N) to line up against the VOL's N=8/N=16.
 * 6. Ratio from H5Dget_storage_size (same as the VOL harness) with the whole
 *    file size reported separately; the old code called H5Fget_filesize before
 *    H5Fclose, so the value could predate the flush.
 * 7. Return codes checked; per-dataset memory guard; create/flush/close/open
 *    timed so the write side includes what actually commits the data.
 *
 * TIMING ASYMMETRY (worth a sentence in the paper): in the filter path compress
 * and I/O are INTERLEAVED per chunk inside H5Dwrite, driven by HDF5. TOTAL is
 * clean; COMPRESS is only obtainable by instrumenting inside the filter
 * callback, and I/O cannot be cleanly separated because HDF5 owns the chunk
 * writes. So the phase columns are left at 0 here. The VOL's clean 3-way split
 * is itself a result.
 *
 * Build:
 *   h5c++ -O2 -std=c++17 bench_filter_timing.cc -o bench_filter_timing
 * ==========================================================================*/
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

/* <<< CONFIRM ON YOUR BUILD: the H5Z filter id registered by robertu94's
 * libpressio filter plugin. There is no well-known number. Override at runtime
 * with LIBPRESSIO_H5Z_FILTER_ID=<n> rather than recompiling. */
#ifndef LIBPRESSIO_H5Z_FILTER_ID_DEFAULT
#define LIBPRESSIO_H5Z_FILTER_ID_DEFAULT 32026   /* placeholder */
#endif

static H5Z_filter_t lp_filter_id(void) {
    const char *e = getenv("LIBPRESSIO_H5Z_FILTER_ID");
    return (H5Z_filter_t)(e && *e ? atoi(e) : LIBPRESSIO_H5Z_FILTER_ID_DEFAULT);
}

typedef enum { FILTER_DEFLATE, FILTER_LIBPRESSIO } filter_backend_t;

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

/* Dataset-creation plist: chunking (mandatory for filters) + codec.
 *
 * chunk_n splits the SLOWEST-varying dimension into n pieces, which is the
 * closest HDF5 analogue to the VOL's flat N-way split. HDF5 tolerates a ragged
 * final chunk but pads it on disk -- a space cost the VOL container does not
 * pay, and worth mentioning when comparing file sizes. */
static hid_t make_dcpl(const bench_dataset_t *d, filter_backend_t backend,
                       const bench_compressor_t *c, int chunk_n,
                       int deflate_level) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, chunk);

    if (chunk_n > 1) {
        hsize_t slow = chunk[0];
        hsize_t per  = (slow + (hsize_t)chunk_n - 1) / (hsize_t)chunk_n;
        if (per < 1) per = 1;
        chunk[0] = per;
    }
    H5Pset_chunk(dcpl, d->rank, chunk);

    /* Match the VOL: no fill-value writes, allocate late. Without this the
     * filter path pays a fill pass the VOL container does not. */
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_LATE);

    if (backend == FILTER_DEFLATE) {
        H5Pset_deflate(dcpl, (unsigned)deflate_level);
    } else {
        /* TODO(plugin API): robertu94's libpressio H5Z filter takes its codec
         * and options through a plugin-specific mechanism -- cd_values encoding
         * or an HDF5 property, see the plugin README. Registering mandatory with
         * no cd_values here means the plugin falls back to its own default
         * codec, which is NOT the same as c->pressio_id. Until this is wired up,
         * --libpressio results are not apples-to-apples and should not be
         * plotted against the VOL.
         *
         * What needs to happen: encode bench_compressor_opts_json(c, d, ...)
         * into whatever the plugin expects, so the SAME codec and the SAME
         * error bound run here as in the VOL. */
        (void)c;
        unsigned int cd_values[1] = {0};
        H5Pset_filter(dcpl, lp_filter_id(), H5Z_FLAG_MANDATORY, 0, cd_values);
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
        return -1;
    }

    /* Same preprocessing as the VOL harness, or the ratios are incomparable. */
    if (d.xform != BENCH_XFORM_NONE)
        bench_apply_xform(hbuf, bench_num_elements(&d), d.dtype, d.xform);

    /* Measured range feeds bench_abs_threshold() for relative-bounded entries. */
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
    const int lossless = (backend == FILTER_DEFLATE) ? 1 : c->lossless;
    const double thr = (backend == FILTER_DEFLATE)
                           ? 0.0 : bench_abs_threshold(c, &d, range);

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
        hid_t dcpl  = make_dcpl(&d, backend, c, chunk_n, deflate_level);
        hid_t dset  = H5Dcreate2(file, d.name, ftype, space, H5P_DEFAULT,
                                 dcpl, H5P_DEFAULT);
        if (dset < 0) {
            std::fprintf(stderr, "[ERR] H5Dcreate2 %s (filter unavailable?)\n", d.name);
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

        std::printf("  %-22s %-12s N=%-3d C=%6.2f W=%9.2f F=%8.2f X=%7.2f | "
                    "O=%6.2f R=%9.2f ms  ratio=%6.2fx  file=%.1f MiB  "
                    "RMSE=%.3e maxae=%.3e\n",
                    d.name, cname, chunk_n, create_ms, write_ms, flush_ms,
                    close_ms, open_ms, read_ms, ratio,
                    filesize / (1024.0 * 1024.0), st.rmse, st.maxae);
        std::fflush(stdout);
    }

cleanup:
    std::free(rbuf);
    std::free(hbuf);
    return rc;
}

int main(int argc, char **argv) {
    const char *h5path   = "bench_filter.h5";
    const char *csvpath  = "results_filter.csv";
    const char *xcsvpath = NULL;
    filter_backend_t backend = FILTER_DEFLATE;
    const char *only_cmp = getenv("BENCH_COMP");
    const char *only_ds  = getenv("BENCH_ONLY");
    int chunk_n = 1, deflate_level = 6;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--deflate"))    backend = FILTER_DEFLATE;
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

    if (backend == FILTER_DEFLATE && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0)
        std::fprintf(stderr, "WARNING: deflate filter not available in this HDF5\n");
    if (backend == FILTER_LIBPRESSIO) {
        if (H5Zfilter_avail(lp_filter_id()) <= 0)
            std::fprintf(stderr,
                "WARNING: libpressio filter id %d not available. Set "
                "HDF5_PLUGIN_PATH and LIBPRESSIO_H5Z_FILTER_ID.\n",
                (int)lp_filter_id());
        std::fprintf(stderr,
            "WARNING: the codec/options plumbing for the libpressio filter is "
            "NOT wired up (see make_dcpl TODO). Results are not comparable to "
            "the VOL until it is.\n");
    }

    bench_datasets_validate();

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
            run_one(d, backend, NULL, h5path, chunk_n, deflate_level, csv, xcsv);
        } else {
            for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
                const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
                if (only_cmp && std::strcmp(only_cmp, c->name)) continue;
                run_one(d, backend, c, h5path, chunk_n, deflate_level, csv, xcsv);
            }
        }
        std::remove(h5path);
    }

    std::fclose(csv);
    std::fclose(xcsv);
    std::fprintf(stderr, "wrote %s and %s\n", csvpath, xcsvpath);
    return 0;
}