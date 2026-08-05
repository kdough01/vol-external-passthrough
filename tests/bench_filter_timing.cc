#include <hdf5.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cerrno>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <hdf5_sz3/H5Z_SZ3.hpp>

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

#include <execinfo.h>
#include <csignal>

static void bench_crash_handler(int sig) {
    void *bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n*** caught signal %d ***\n", sig);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    _exit(128 + sig);
}

/* ------------------------------------------------------------------------
 * DURABILITY INSTRUMENTATION -- must match bench_vol_timing.cpp exactly, or
 * the filter and VOL numbers are not comparable.
 *
 * H5Dwrite reaches HDF5's cache; H5Fflush reaches the OS page cache. Only
 * fsync() puts the bytes on the device. Without it, write_ms measures memory
 * and the filter would appear far faster than the VOL, which does fsync.
 *
 * Wall clock, not CPU clock: fsync blocks in the kernel waiting on device
 * completion, which a CPU-time clock does not see at all.
 * --------------------------------------------------------------------- */
static inline double bench_wall_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec * 1.0e-6;
}

struct BenchWallTimer {
    double t0;
    void   start()   { t0 = bench_wall_now_ms(); }
    double stop_ms() { return bench_wall_now_ms() - t0; }
};

static int bench_env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    return std::atoi(s);
}
static int bench_do_fsync(void)     { return bench_env_int("BENCH_FSYNC", 1) != 0; }
static int bench_do_dropcache(void) { return bench_env_int("BENCH_DROP_CACHE", 1) != 0; }
static int bench_do_syncdir(void)   { return bench_env_int("BENCH_SYNC_DIR", 1) != 0; }

/* fsync the file's data to the device. A second read-only descriptor is fine:
 * fsync acts on the inode, not on the descriptor's write history, so it
 * flushes everything HDF5 wrote through its own fd. sync_dir also fsyncs the
 * containing directory so the newly created dirent is durable. */
static double bench_fsync_path(const char *path, int sync_dir) {
    BenchWallTimer t; t.start();

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[warn] fsync open failed %s: %s\n",
                     path, std::strerror(errno));
        return 0.0;
    }
    if (::fsync(fd) != 0)
        std::fprintf(stderr, "[warn] fsync failed %s: %s\n",
                     path, std::strerror(errno));
    ::close(fd);

    if (sync_dir) {
        char dir[1024];
        std::snprintf(dir, sizeof(dir), "%s", path);
        char *slash = std::strrchr(dir, '/');
        if (slash) { if (slash == dir) dir[1] = '\0'; else *slash = '\0'; }
        else       { std::snprintf(dir, sizeof(dir), "."); }
        int dfd = ::open(dir, O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) { (void)::fsync(dfd); ::close(dfd); }
    }
    return t.stop_ms();
}

/* Evict the file from the page cache so the read phase touches the device.
 * POSIX_FADV_DONTNEED only drops CLEAN pages, hence the fsync first. Best
 * effort: on Lustre/GPFS, client-side caching is filesystem-managed and some
 * pages may survive. */
static double bench_evict_path(const char *path) {
    BenchWallTimer t; t.start();

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return 0.0;
    (void)::fsync(fd);
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0)
        std::fprintf(stderr, "[warn] fadvise DONTNEED failed %s: %s\n",
                     path, std::strerror(errno));
    ::close(fd);

    return t.stop_ms();
}

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

/* Identical to bench_vol_timing.cpp's XCSV_HEADER so the two merge directly.
 * 21 fields: sync_ms / csync_ms / evict_ms were added with the durability
 * instrumentation and are inserted AFTER flush_ms, so any consumer indexing
 * by position must be updated (abs_thresh=19, maxae=20, bound_ok=21). */
static const char *XCSV_HEADER =
    "dataset,compressor,codec_kind,chunk_n,rep,"
    "logical_bytes,stored_bytes,ratio,"
    "create_ms,write_ms,flush_ms,sync_ms,close_ms,csync_ms,"
    "evict_ms,open_ms,read_ms,"
    "rmse,abs_thresh,maxae,bound_ok\n";

/* A rep=-1 failure row, in the same 21-field layout. */
static void xcsv_failure_row(FILE *xcsv, const char *dataset, const char *comp,
                             const char *kind, int chunk_n,
                             unsigned long long logical) {
    std::fprintf(xcsv,
        "%s,%s,%s,%d,-1,%llu,0,0,"
        "-1,-1,-1,-1,-1,-1,"
        "-1,-1,-1,"
        "-1,-1,-1,0\n",
        dataset, comp, kind, chunk_n, logical);
    std::fflush(xcsv);
}

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

    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);

    hsize_t chunk[BENCH_MAX_RANK];
    for (int i = 0; i < d->rank; ++i) chunk[i] = dims[i];

    /* Slowest-first multi-dimensional split. `remaining` is how many pieces we
     * still owe; each dimension absorbs as many as it can (capped by its extent)
     * before the remainder spills onto the next-faster dimension. */
    if (chunk_n > 1) {
        hsize_t remaining = (hsize_t)chunk_n;
        for (int i = 0; i < d->rank && remaining > 1; ++i) {
            hsize_t split = (dims[i] < remaining) ? dims[i] : remaining;
            if (split < 1) split = 1;
            chunk[i]  = (dims[i] + split - 1) / split;          /* ceil */
            if (chunk[i] < 1) chunk[i] = 1;
            remaining = (remaining + split - 1) / split;        /* ceil */
        }
    }
    H5Pset_chunk(dcpl, d->rank, chunk);

    /* Report the ACTUAL chunk geometry (bytes are what the plot needs). */
    {
        size_t itemsz = (d->dtype == BENCH_F64) ? 8u : 4u;
        unsigned long long celems = 1ULL, nch = 1ULL;
        for (int i = 0; i < d->rank; ++i) {
            celems *= (unsigned long long)chunk[i];
            nch    *= (unsigned long long)((dims[i] + chunk[i] - 1) / chunk[i]);
        }
        unsigned long long cbytes = celems * (unsigned long long)itemsz;
        std::fprintf(stdout,
            "CHUNKINFO dataset=%s chunk_n=%d nchunks_actual=%llu "
            "chunk_elems=%llu chunk_bytes=%llu\n",
            d->name, chunk_n, nch, celems, cbytes);
        std::fflush(stdout);
        const char *cipath = std::getenv("BENCH_CHUNKINFO");
        if (cipath && *cipath) {
            FILE *cf = std::fopen(cipath, "a");
            if (cf) {
                std::fprintf(cf, "%s,%d,%llu,%llu,%llu\n",
                             d->name, chunk_n, nch, celems, cbytes);
                std::fclose(cf);
            }
        }
    }

    /* Match the VOL: no fill-value writes, allocate late. */
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_LATE);

    switch (backend) {
    case FILTER_DEFLATE:
        H5Pset_deflate(dcpl, (unsigned)deflate_level);
        break;

    case FILTER_BZIP2: {
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
        std::vector<size_t> cdims;
        for (int i = 0; i < d->rank; ++i)
            cdims.push_back((size_t)chunk[i]);

        SZ3::Config conf;
        conf.setDims(cdims.begin(), cdims.end());
        conf.errorBoundMode = SZ3::EB_ABS;
        conf.absErrorBound  = abs_bound;

        if (H5Pset_filter(dcpl, H5Z_FILTER_SZ3, H5Z_FLAG_MANDATORY, 0, NULL) < 0)
            std::fprintf(stderr, "WARNING: H5Pset_filter(SZ3) failed\n");
        if (set_SZ3_conf_to_H5(dcpl, conf) < 0)
            std::fprintf(stderr, "WARNING: set_SZ3_conf_to_H5 failed\n");
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
            "NOT accuracy=%g.\n", abs_bound);
#endif
        break;
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
        xcsv_failure_row(xcsv, din->name,
                         (backend == FILTER_DEFLATE) ? "deflate" : c->name,
                         "cpu", chunk_n, 0ULL);
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

    const int do_fsync = bench_do_fsync();
    const int do_evict = bench_do_dropcache();
    const int do_sdir  = bench_do_syncdir();

    double create_ms = 0, write_ms = 0, flush_ms = 0, sync_ms = 0;
    double close_ms = 0, csync_ms = 0, evict_ms = 0;
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

        /* H5Fflush only reaches the OS page cache; the fsync below is what
         * forces the bytes to the device. Identical treatment to the VOL. */
        BenchWallTimer ft; ft.start();
        (void)H5Fflush(file, H5F_SCOPE_LOCAL);
        flush_ms = ft.stop_ms();

        if (do_fsync) sync_ms = bench_fsync_path(h5path, do_sdir);

        stored = (unsigned long long)H5Dget_storage_size(dset);  /* == VOL metric */
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(space);

        BenchWallTimer clt; clt.start();
        H5Fclose(file);
        close_ms = clt.stop_ms();

        /* H5Fclose can emit superblock/metadata updates after the flush, so the
         * file is not durable until this second fsync returns. */
        if (do_fsync) csync_ms = bench_fsync_path(h5path, do_sdir);

        if (wret < 0) { std::fprintf(stderr, "[ERR] H5Dwrite %s\n", d.name); rc = -1; goto cleanup; }
    }

    /* ---------------- READ (cold cache) ---------------- */
    {
        std::memset(rbuf, 0, raw);

        /* Not counted in the read total: cache teardown, not read cost. */
        if (do_evict) evict_ms = bench_evict_path(h5path);

        BenchWallTimer ot; ot.start();
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
        bench_csv_row(csv, d.name, cname, "filter", "write", "sync",  sync_ms, -1.0, -1.0);
        bench_csv_row(csv, d.name, cname, "filter", "write", "csync", csync_ms, -1.0, -1.0);
        bench_csv_row(csv, d.name, cname, "filter", "read",  "evict", evict_ms, -1.0, -1.0);
        bench_csv_row(csv, d.name, cname, "filter", "read",  "total", read_ms, -1.0, st.rmse);

        std::fprintf(xcsv,
            "%s,%s,%s,%d,%d,%llu,%llu,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,"
            "%.6e,%.6e,%.6e,%d\n",
            d.name, cname, kind, chunk_n, 0,
            (unsigned long long)raw, stored, ratio,
            create_ms, write_ms, flush_ms, sync_ms, close_ms, csync_ms,
            evict_ms, open_ms, read_ms,
            st.rmse, thr, st.maxae, bound_ok);
        std::fflush(xcsv);

        std::printf("  %-14s %-10s N=%-3d C=%6.2f W=%9.2f F=%8.2f S=%8.2f "
                    "X=%7.2f S2=%7.2f | E=%6.2f O=%6.2f R=%9.2f ms  "
                    "ratio=%6.2fx  file=%.1f MiB  RMSE=%.3e maxae=%.3e\n",
                    d.name, cname, chunk_n, create_ms, write_ms, flush_ms,
                    sync_ms, close_ms, csync_ms, evict_ms, open_ms, read_ms,
                    ratio, filesize / (1024.0 * 1024.0), st.rmse, st.maxae);
        std::fflush(stdout);
    }

cleanup:
    /* Record failures as a rep=-1 row rather than leaving a header-only CSV. An
     * unsupported configuration IS a result: einspline37 as a single chunk
     * exceeds HDF5's 4 GiB limit, which the VOL container has no equivalent of. */
    if (rc != 0) {
        xcsv_failure_row(xcsv, d.name, cname, kind, chunk_n,
                         (unsigned long long)raw);
        std::fprintf(stderr, "[RECORDED FAILURE] %s/%s N=%d\n", d.name, cname, chunk_n);
    }
    std::free(rbuf);
    std::free(hbuf);
    return rc;
}

int main(int argc, char **argv) {
    signal(SIGSEGV, bench_crash_handler);
    signal(SIGABRT, bench_crash_handler);
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
    std::fprintf(stderr,
        "[filter] durability: fsync=%d sync_dir=%d drop_cache=%d "
        "(BENCH_FSYNC / BENCH_SYNC_DIR / BENCH_DROP_CACHE) -- these MUST match "
        "the VOL run or the two are not comparable\n",
        bench_do_fsync(), bench_do_syncdir(), bench_do_dropcache());
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