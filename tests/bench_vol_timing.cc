#include <hdf5.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cerrno>
#include <stdexcept>
#include <exception>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define BENCH_CONFIG_ENABLE_HDF5
#include "bench_config.h"
#include "bench_timing.h"
#include "miranda.h"

static const char *XCSV_HEADER =
    "dataset,compressor,codec_kind,chunk_n,rep,"
    "logical_bytes,stored_bytes,ratio,"
    "create_ms,write_ms,flush_ms,sync_ms,close_ms,csync_ms,"
    "evict_ms,open_ms,read_ms,"
    "rmse,abs_thresh,maxae,bound_ok\n";

typedef struct {
    double create_ms, write_ms, flush_ms, sync_ms, close_ms, csync_ms;
    double evict_ms, open_ms, read_ms;
    unsigned long long stored;
} rep_timing;

/* ------------------------------------------------------------------ *
 * Wall-clock timer.  fsync() spends its time blocked in the kernel on
 * device completion, which a CPU-time clock will not see at all, so the
 * durability phases must be measured against CLOCK_MONOTONIC.
 * ------------------------------------------------------------------ */
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

static void xcsv_row(FILE *fp, const char *dset, const bench_compressor_t *c,
                     int chunk_n, int rep, unsigned long long logical,
                     const rep_timing *t,
                     double rmse, double abs_thresh, double maxae) {
    if (!fp) return;
    const double ratio = t->stored ? (double)logical / (double)t->stored : 0.0;
    /* -1 == "not applicable": rate-mode entries are lossy with no error bound,
     * so a pass/fail verdict would be meaningless.  Filter these out downstream
     * rather than reading them as failures. */
    const int bound_ok = !bench_bound_is_checkable(c) ? -1
                       : (abs_thresh > 0.0) ? (maxae <= 2.0 * abs_thresh)
                                            : (maxae == 0.0);
    std::fprintf(fp,
        "%s,%s,%s,%d,%d,%llu,%llu,%.4f,"
        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
        "%.4f,%.4f,%.4f,"
        "%.6e,%.6e,%.6e,%d\n",
        dset, c->name, bench_codec_kind_name(c->kind), chunk_n, rep,
        logical, t->stored, ratio,
        t->create_ms, t->write_ms, t->flush_ms, t->sync_ms,
        t->close_ms, t->csync_ms,
        t->evict_ms, t->open_ms, t->read_ms,
        rmse, abs_thresh, maxae, bound_ok);
    std::fflush(fp);
}

typedef struct { double min, max, mean, rmse, maxae; } bench_stats;

static bench_stats bench_compute_stats(const void *orig, const void *dec,
                                       size_t nelem, bench_dtype_t dt,
                                       bench_xform_t xf) {
    double mn = DBL_MAX, mx = -DBL_MAX, sum = 0.0, se = 0.0, maxae = 0.0;
    for (size_t i = 0; i < nelem; ++i) {
        double o = (dt == BENCH_F64) ? ((const double *)orig)[i]
                                     : (double)((const float *)orig)[i];
        double v = (dt == BENCH_F64) ? ((const double *)dec)[i]
                                     : (double)((const float *)dec)[i];
        /* Invert preprocessing so error is in physical units. NONE = identity. */
        if      (xf == BENCH_XFORM_LOG1P) { o = expm1(o); v = expm1(v); }
        else if (xf == BENCH_XFORM_ASINH) { o = sinh(o);  v = sinh(v);  }
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

static void bench_zero_run(const void *orig, const void *dec,
                           size_t nelem, bench_dtype_t dt, const char *label) {
    size_t total = 0, run = 0, best = 0, best_start = 0, cur = 0;
    for (size_t i = 0; i < nelem; ++i) {
        double o = (dt==BENCH_F64)? ((const double*)orig)[i] : (double)((const float*)orig)[i];
        double v = (dt==BENCH_F64)? ((const double*)dec )[i] : (double)((const float*)dec )[i];
        if (v == 0.0 && o != 0.0) {
            if (run == 0) cur = i;
            if (++run > best) { best = run; best_start = cur; }
            ++total;
        } else run = 0;
    }
    std::fprintf(stderr,
        "[zerorun] %-28s fill_zeros=%zu (%.2f%%)  longest=%zu @ [%zu..%zu]\n",
        label, total, 100.0*total/nelem, best, best_start, best_start + best);
}

/* thr is an ABSOLUTE error threshold from bench_abs_threshold(), not the
 * dataset's nominal relative bound. */
static int bench_check_chunks(const void *orig, const void *dec, size_t nelem,
                              bench_dtype_t dt, int nchunks, double thr,
                              const char *label) {
    if (nchunks < 1) nchunks = 1;
    size_t base = nelem / (size_t)nchunks, rem = nelem % (size_t)nchunks, off = 0;
    int bad = 0;
    for (int c = 0; c < nchunks; ++c) {
        size_t cn = base + ((size_t)c < rem ? 1 : 0);
        double se = 0.0, maxae = 0.0; size_t nz = 0;
        for (size_t i = off; i < off + cn; ++i) {
            double o = (dt==BENCH_F64)? ((const double*)orig)[i] : (double)((const float*)orig)[i];
            double v = (dt==BENCH_F64)? ((const double*)dec )[i] : (double)((const float*)dec )[i];
            double e = o - v, ae = std::fabs(e);
            se += e*e; if (ae > maxae) maxae = ae;
            if (v == 0.0 && std::fabs(o) > thr) ++nz;
        }
        int chunk_bad = (maxae > 2.0 * thr);
        bad += chunk_bad;
        std::fprintf(stderr,
            "[chunkchk] %-28s chunk %2d/%-2d n=%zu rmse=%.3e maxae=%.3e "
            "thr=%.3e fillzero=%zu %s\n",
            label, c, nchunks, cn, std::sqrt(se/cn), maxae, thr, nz,
            chunk_bad ? "<-- BAD" : "");
        off += cn;
    }
    if (bad) std::fprintf(stderr, "[chunkchk] %-28s **%d/%d chunks bad**\n", label, bad, nchunks);
    return bad;
}

static int bench_locate_bad(const void *orig, const void *dec, size_t nelem,
                            bench_dtype_t dt, int nchunks, double thr_in,
                            const char *label) {
    if (nchunks < 1) nchunks = 1;
    const size_t dsize = bench_dtype_size(dt);
    const size_t base  = nelem / (size_t)nchunks;
    const size_t rem   = nelem % (size_t)nchunks;
    const double thr   = 2.0 * thr_in;
    const size_t GAP   = 256;   /* merge bad runs split by < GAP good elems */

    auto chunk_of = [&](size_t i, size_t *cstart) -> int {
        size_t big = rem * (base + 1);
        if (i < big) { int c = (int)(i / (base + 1)); *cstart = (size_t)c * (base + 1); return c; }
        size_t j = i - big; int c = (int)rem + (int)(j / base);
        *cstart = big + (j - j % base); return c;
    };
    auto val = [&](const void *p, size_t i) {
        return (dt == BENCH_F64) ? ((const double*)p)[i] : (double)((const float*)p)[i];
    };

    int nblocks = 0, printed = 0;
    size_t i = 0;
    while (i < nelem) {
        if (std::fabs(val(orig, i) - val(dec, i)) <= thr) { ++i; continue; }
        size_t start = i, last_bad = i, nbad = 0; double blk_maxae = 0.0;
        while (i < nelem) {
            double ae = std::fabs(val(orig, i) - val(dec, i));
            if (ae > thr) { last_bad = i; ++nbad; if (ae > blk_maxae) blk_maxae = ae; }
            else if (i - last_bad > GAP) break;
            ++i;
        }
        size_t len = last_bad - start + 1, cstart; int c = chunk_of(start, &cstart);
        size_t off = start - cstart;
        ++nblocks;
        if (printed++ < 40)
            std::fprintf(stderr,
              "[badblk] %-24s blk %d: elem[%zu..%zu] len=%zu nbad=%zu maxae=%.3e | "
              "byte_off=%zu | chunk %d (starts elem %zu) off_in_chunk=%zu elem / %zu B\n",
              label, nblocks, start, last_bad, len, nbad, blk_maxae,
              start * dsize, c, cstart, off, off * dsize);
    }
    std::fprintf(stderr, nblocks ? "[badblk] %-24s %d bad block(s)\n"
                                 : "[badblk] %-24s clean\n", label, nblocks);
    return nblocks;
}

static int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    return std::atoi(s);
}
static int bench_verify(void)  { return env_int("BENCH_VERIFY", 0) != 0; }
static int bench_keep_h5(void) { return env_int("BENCH_KEEP_H5", 0) != 0; }

/* Durability / cache controls.  Both default ON: without them write_ms and
 * read_ms measure the page cache, not the storage device, which is what makes
 * the io_ms column jump around between runs.
 *   BENCH_FSYNC=0       -- skip fsync (old, cache-only behaviour)
 *   BENCH_DROP_CACHE=0  -- skip page-cache eviction before the read phase
 *   BENCH_SYNC_DIR=0    -- skip the parent-directory fsync
 */
static int bench_do_fsync(void)      { return env_int("BENCH_FSYNC", 1) != 0; }
static int bench_do_dropcache(void)  { return env_int("BENCH_DROP_CACHE", 1) != 0; }
static int bench_do_syncdir(void)    { return env_int("BENCH_SYNC_DIR", 1) != 0; }

/* fsync the file's data to the device.  Opening a second read-only descriptor
 * is fine: fsync() acts on the inode, not on the descriptor's write history,
 * so it flushes everything HDF5 wrote through its own fd.  sync_dir also
 * fsyncs the containing directory so the newly created dirent is durable. */
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

/* Evict the file from the page cache so the read phase actually touches the
 * device.  POSIX_FADV_DONTNEED only drops CLEAN pages, hence the fsync first.
 * Best effort: on a shared node, or on Lustre/GPFS where client-side caching
 * is managed by the filesystem, some pages may survive. */
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

static int name_selected(const char *sel, const char *name) {
    if (!sel || !*sel) return 1;
    size_t nl = std::strlen(name);
    for (const char *p = std::strstr(sel, name); p; p = std::strstr(p + 1, name)) {
        char b = (p == sel) ? ',' : p[-1], a = p[nl];
        if ((b == ',' || b == ' ') && (a == ',' || a == ' ' || a == '\0')) return 1;
    }
    return 0;
}

/* "<base>.h5" + suffix -> "<base><suffix>.h5"; no extension -> plain append. */
static void derive_path(const char *base, const char *suffix,
                        char *out, size_t n) {
    const char *dot = std::strrchr(base, '.');
    if (dot && 0 == std::strcmp(dot, ".h5"))
        std::snprintf(out, n, "%.*s%s.h5", (int)(dot - base), base, suffix);
    else
        std::snprintf(out, n, "%s%s.h5", base, suffix);
}

typedef struct {
    double             write_ms, flush_ms, sync_ms, close_ms, csync_ms;
    double             create_ms, evict_ms, open_ms, read_ms;
    size_t             raw_bytes;
    unsigned long long storage;
    int                n;
} bench_file_acc;

static int run_rep(const char *path, const char *dsname,
                   const bench_dataset_t *d, const bench_compressor_t *c,
                   const char *oj, hid_t space, hid_t ntype,
                   const void *hbuf, void *rbuf, size_t raw_bytes,
                   rep_timing *t) {
    std::memset(t, 0, sizeof(*t));

    const int do_fsync = bench_do_fsync();
    const int do_evict = bench_do_dropcache();
    const int do_sdir  = bench_do_syncdir();

    /* ---------------- WRITE PHASE (own file) ---------------- */
    {
        BenchCpuTimer ct; ct.start();
        hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        t->create_ms = ct.stop_ms();
        if (file < 0) {
            std::fprintf(stderr, "[ERR] H5Fcreate failed %s\n", path);
            return -1;
        }

        hid_t dcpl = make_dcpl(c->pressio_id, oj);
        hid_t dset = H5Dcreate2(file, dsname, ntype, space, H5P_DEFAULT,
                                dcpl, H5P_DEFAULT);
        if (dset < 0) {
            std::fprintf(stderr, "[ERR] H5Dcreate2 failed %s\n", dsname);
            H5Pclose(dcpl); H5Fclose(file);
            return -1;
        }

        BenchCpuTimer wt; wt.start();
        herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
        t->write_ms = wt.stop_ms();

        /* H5Dwrite only reaches HDF5's cache; H5Fflush only reaches the OS
         * page cache.  The fsync below is what actually forces the bytes to
         * the device, and it is the phase that makes io_ms reproducible. */
        BenchWallTimer ft; ft.start();
        (void)H5Fflush(file, H5F_SCOPE_LOCAL);
        t->flush_ms = ft.stop_ms();

        if (do_fsync) t->sync_ms = bench_fsync_path(path, do_sdir);

        t->stored = (unsigned long long)H5Dget_storage_size(dset);
        H5Dclose(dset);
        H5Pclose(dcpl);

        BenchWallTimer clt; clt.start();
        H5Fclose(file);
        t->close_ms = clt.stop_ms();

        /* H5Fclose can emit superblock/metadata updates after the flush, so
         * the file is not durable until this second fsync returns.  It should
         * be small; if it is not, HDF5 deferred real payload to close. */
        if (do_fsync) t->csync_ms = bench_fsync_path(path, do_sdir);

        if (wret < 0) {
            std::fprintf(stderr, "[ERR] H5Dwrite failed %s\n", dsname);
            return -1;
        }
    }

    /* ---------------- READ PHASE (fresh open, cold cache) ---------------- */
    {
        std::memset(rbuf, 0, raw_bytes);

        /* Not counted in read_total: this is cache teardown, not read cost. */
        if (do_evict) t->evict_ms = bench_evict_path(path);

        BenchWallTimer ot; ot.start();
        hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        t->open_ms = ot.stop_ms();
        if (file < 0) {
            std::fprintf(stderr, "[ERR] H5Fopen failed %s\n", path);
            return -1;
        }

        hid_t dset = H5Dopen2(file, dsname, H5P_DEFAULT);
        if (dset < 0) {
            std::fprintf(stderr, "[ERR] H5Dopen2 failed %s\n", dsname);
            H5Fclose(file);
            return -1;
        }

        BenchCpuTimer rt; rt.start();
        herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        t->read_ms = rt.stop_ms();

        H5Dclose(dset);
        H5Fclose(file);

        if (rret < 0) {
            std::fprintf(stderr, "[ERR] H5Dread failed %s\n", dsname);
            return -1;
        }
    }
    return 0;
}

static void run_pair(const char *h5base,
                     const bench_dataset_t *d, const bench_compressor_t *c,
                     const void *hbuf, const void *wbuf, void *rbuf,
                     size_t raw_bytes, double measured_range,
                     FILE *csv, FILE *xcsv, bench_file_acc *acc) {
    const bool   dbg     = std::getenv("BENCH_DEBUG") != NULL;
    const int    verify  = bench_verify();
    const int    chunk_n = bench_effective_chunk_n(c);
    const double thr     = bench_abs_threshold(c, d, measured_range);

    hid_t space = H5I_INVALID_HID;

  try {
    size_t  nelem = bench_num_elements(d);
    hid_t   ntype = bench_dataset_h5native(d);
    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    space = H5Screate_simple(d->rank, dims, NULL);

    char opts[256];
    bench_compressor_opts_json(c, d, opts, sizeof(opts));
    const char *oj = (std::strcmp(opts, "{}") == 0) ? NULL : opts;

    if (dbg)
        std::fprintf(stderr,
            "[dbg run_pair] %s_%s rank=%d nelem=%zu xform=%s pressio_id=%s "
            "chunk_n=%d abs_thresh=%.3e opts=%s\n",
            d->name, c->name, d->rank, nelem, bench_xform_name(d->xform),
            c->pressio_id ? c->pressio_id : "(null)", chunk_n, thr,
            oj ? oj : "(default)");

    {
        char       path[1024], suffix[64], dsname[224];
        rep_timing t;

        std::snprintf(suffix, sizeof(suffix), "_%s", c->name);
        derive_path(h5base, suffix, path, sizeof(path));
        std::snprintf(dsname, sizeof(dsname), "%s_%s", d->name, c->name);

        if (run_rep(path, dsname, d, c, oj, space, ntype,
                    wbuf, rbuf, raw_bytes, &t) != 0) {
            if (!bench_keep_h5()) std::remove(path);
            if (space != H5I_INVALID_HID) H5Sclose(space);
            return;
        }

        /* ---------------- FIDELITY ---------------- */
        bench_stats st = bench_compute_stats(hbuf, rbuf, nelem, d->dtype,
                                             BENCH_XFORM_NONE);

        if (verify) {
            bench_zero_run(hbuf, rbuf, nelem, d->dtype, dsname);
            /* thr == 0 on a lossy rate-mode entry would flag every chunk bad,
             * so run the threshold-based checks only where a bound exists. */
            if (bench_bound_is_checkable(c)) {
                bench_check_chunks(hbuf, rbuf, nelem, d->dtype, chunk_n, thr, dsname);
                bench_locate_bad(hbuf, rbuf, nelem, d->dtype, chunk_n, thr, dsname);
            } else {
                std::fprintf(stderr, "[chunkchk] %-28s skipped (no error bound; "
                             "rate mode)\n", dsname);
            }
        }

        if (c->lossless && st.maxae != 0.0)
            std::fprintf(stderr, "[FAIL] %-28s declared lossless but maxae=%.6e\n",
                         dsname, st.maxae);
        else if (thr > 0.0 && st.maxae > 2.0 * thr)
            std::fprintf(stderr, "[FAIL] %-28s maxae=%.6e exceeds 2x configured "
                                 "bound %.6e\n", dsname, st.maxae, thr);

        const double ratio = t.stored ? (double)raw_bytes / (double)t.stored : 0.0;

        bench_csv_row(csv, d->name, c->name, "vol", "write", "total", t.write_ms, ratio, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "write", "flush", t.flush_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "write", "sync",  t.sync_ms,  -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "write", "close", t.close_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "write", "csync", t.csync_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "read",  "evict", t.evict_ms, -1.0, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "read",  "total", t.read_ms, -1.0, st.rmse);

        xcsv_row(xcsv, d->name, c, chunk_n, 0,
                 (unsigned long long)raw_bytes, &t, st.rmse, thr, st.maxae);

        if (acc) {
            acc->create_ms += t.create_ms; acc->write_ms += t.write_ms;
            acc->flush_ms  += t.flush_ms;  acc->sync_ms  += t.sync_ms;
            acc->close_ms  += t.close_ms;  acc->csync_ms += t.csync_ms;
            acc->evict_ms  += t.evict_ms;
            acc->open_ms   += t.open_ms;   acc->read_ms  += t.read_ms;
            acc->raw_bytes += raw_bytes;   acc->storage  += t.stored;
            acc->n         += 1;
        }

        std::printf("  %-28s C=%7.2f W=%9.2f F=%8.2f S=%9.2f X=%8.2f S2=%7.2f | "
                    "E=%7.2f O=%7.2f R=%9.2f ms  ratio=%6.2fx  "
                    "RMSE=%.3e maxae=%.3e%s\n",
                    dsname, t.create_ms, t.write_ms, t.flush_ms, t.sync_ms,
                    t.close_ms, t.csync_ms,
                    t.evict_ms, t.open_ms, t.read_ms, ratio, st.rmse, st.maxae,
                    (d->xform != BENCH_XFORM_NONE) ? " (log-space)" : "");
        std::fflush(stdout);

        if (!bench_keep_h5()) std::remove(path);
    }

  } catch (const std::exception &e) {
      std::fprintf(stderr, "[ERR] run_pair %s/%s threw: %s (skipped; sweep continues)\n",
                   d->name, c->name, e.what());
      std::fflush(stderr);
  } catch (...) {
      std::fprintf(stderr, "[ERR] run_pair %s/%s threw unknown exception (skipped)\n",
                   d->name, c->name);
      std::fflush(stderr);
  }

    if (space != H5I_INVALID_HID) H5Sclose(space);
}

int main(int argc, char **argv) {
    const char *h5base   = (argc > 1) ? argv[1] : "bench_out.h5";
    const char *csvpath  = (argc > 2) ? argv[2] : "results_vol.csv";
    const char *xcsvpath = (argc > 3) ? argv[3] : std::getenv("BENCH_CSV_EX");
    const char *only     = std::getenv("BENCH_ONLY");
    const char *only_cmp = std::getenv("BENCH_COMP");
    const bool  dbg      = std::getenv("BENCH_DEBUG") != NULL;

    char xcsv_default[512];
    if (!xcsvpath || !*xcsvpath) {
        std::snprintf(xcsv_default, sizeof(xcsv_default), "%s.ext.csv", csvpath);
        xcsvpath = xcsv_default;
    }

    std::fprintf(stderr,
        "[dbg main] h5base=%s csv=%s xcsv=%s only=%s comp=%s "
        "verify=%d keep_h5=%d debug=%d\n",
        h5base, csvpath, xcsvpath, only ? only : "(all)",
        only_cmp ? only_cmp : "(all)",
        bench_verify(), bench_keep_h5(), (int)dbg);
    std::fprintf(stderr,
        "[dbg main] durability: fsync=%d sync_dir=%d drop_cache=%d "
        "(BENCH_FSYNC / BENCH_SYNC_DIR / BENCH_DROP_CACHE)\n",
        bench_do_fsync(), bench_do_syncdir(), bench_do_dropcache());
    std::fprintf(stderr,
        "[dbg main] one measurement per (dataset x compressor), no warmup, "
        "own file at <base>_<comp>.h5\n");

    register_vol_properties();
    bench_datasets_validate();

    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    FILE *xcsv = std::fopen(xcsvpath, "w");
    if (!xcsv) { std::perror("xcsv"); std::fclose(csv); return 1; }
    std::fputs(XCSV_HEADER, xcsv);

    bench_file_acc acc;
    std::memset(&acc, 0, sizeof(acc));
    int n_ok = 0, n_skip = 0;

    const size_t mem_avail = bench_mem_available_bytes();

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (!name_selected(only, d->name)) continue;
        if (access(d->path, R_OK) != 0) {
            std::fprintf(stderr, "[dbg main] skip %s (unreadable: %s)\n",
                         d->name, d->path);
            n_skip++; continue;
        }

        if (d->src != BENCH_SRC_HDF5 && mem_avail) {
            size_t need = 2 * bench_num_bytes(d);
            if (need > (size_t)(0.9 * (double)mem_avail)) {
                std::fprintf(stderr,
                    "[SKIP] %s needs %.1f GiB for hbuf+rbuf but only %.1f GiB "
                    "available -- raise the job's mem= request\n",
                    d->name, bench_gib(need), bench_gib(mem_avail));
                n_skip++; continue;
            }
        }

        bench_dataset_t rz;
        size_t raw = 0;
        void *hbuf = bench_load_field(d, &rz, &raw);
        if (!hbuf) {
            std::fprintf(stderr, "[ERR] load failed %s\n", d->name);
            n_skip++; continue;
        }

        if (rz.xform != BENCH_XFORM_NONE) {
            if (dbg) std::fprintf(stderr, "[dbg main] applying %s to %s\n",
                                  bench_xform_name(rz.xform), rz.name);
            bench_apply_xform(hbuf, bench_num_elements(&rz), rz.dtype, rz.xform);
        }

        double measured_range = 0.0;
        {
            size_t ne = bench_num_elements(&rz);
            double mn = DBL_MAX, mx = -DBL_MAX;
            for (size_t i = 0; i < ne; ++i) {
                double v = (rz.dtype == BENCH_F64) ? ((const double*)hbuf)[i]
                                                   : (double)((const float*)hbuf)[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            measured_range = mx - mn;
            std::fprintf(stderr,
                         "RANGE %-14s min=%.6e max=%.6e range=%.6e xform=%s\n",
                         rz.name, mn, mx, measured_range,
                         bench_xform_name(rz.xform));
        }

        void *rbuf = std::malloc(raw);
        if (!rbuf) {
            std::fprintf(stderr, "[ERR] OOM rbuf %s\n", rz.name);
            std::free(hbuf);
            continue;
        }

        const void *wbuf = hbuf;
#ifdef USE_CUDA
        void *dbuf = NULL;      /* the application's device-resident field */
        void *sbuf = NULL;      /* host copy the filter path would require */
        const int dev_mode  = (std::getenv("BENCH_DEVICE_INPUT") != NULL);
        const int host_mode = (std::getenv("BENCH_HOST_STAGED")  != NULL);

        if (dev_mode || host_mode) {
            if (cudaMalloc(&dbuf, raw) != cudaSuccess) {
                std::fprintf(stderr, "[ERR] cudaMalloc %.1f MiB failed for %s\n",
                             raw / (1024.0 * 1024.0), rz.name);
                std::free(rbuf); std::free(hbuf); continue;
            }
            cudaMemcpy(dbuf, hbuf, raw, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            std::fprintf(stderr, "[device] %s: %.1f MiB resident on GPU\n",
                         rz.name, raw / (1024.0 * 1024.0));
        }

        if (dev_mode) {
            wbuf = dbuf;
            std::fprintf(stderr, "[device] arm=device: VOL receives the device "
                                 "pointer; no host round trip for the write\n");
        } else if (host_mode) {
            sbuf = std::malloc(raw);
            if (!sbuf) {
                std::fprintf(stderr, "[ERR] OOM staging %zu bytes\n", raw);
                cudaFree(dbuf); std::free(rbuf); std::free(hbuf); continue;
            }
            cudaEvent_t e0, e1; float d2h_ms = 0.0f;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            cudaMemcpy(sbuf, dbuf, raw, cudaMemcpyDeviceToHost);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&d2h_ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            wbuf = sbuf;
            std::fprintf(stderr,
                         "[device] arm=host d2h_ms=%.3f pcie_in=%.1f MiB\n",
                         (double)d2h_ms, raw / (1024.0 * 1024.0));
        }
#endif

        std::printf("\n=== %s (%.1f MiB, %s, %s%s) ===\n", rz.name,
                    raw / (1024.0 * 1024.0), bench_dtype_name(rz.dtype),
                    bench_src_name(rz.src),
                    rz.xform != BENCH_XFORM_NONE ? ", transformed" : "");

        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (!name_selected(only_cmp, c->name)) continue;
            run_pair(h5base, &rz, c, hbuf, wbuf, rbuf, raw,
                     measured_range, csv, xcsv, &acc);
        }

#ifdef USE_CUDA
        if (dbuf) cudaFree(dbuf);
        if (sbuf) std::free(sbuf);
#endif
        std::free(rbuf);
        std::free(hbuf);
        n_ok++;
    }

    const double file_wtotal = acc.create_ms + acc.write_ms + acc.flush_ms
                             + acc.sync_ms  + acc.close_ms + acc.csync_ms;
    const double file_rtotal = acc.open_ms + acc.read_ms;   /* evict excluded */
    const double file_ratio  = acc.storage ? (double)acc.raw_bytes / (double)acc.storage : 0.0;

    std::printf("\n=== TOTALS over %d measurement files: create=%.2f write=%.2f "
                "flush=%.2f sync=%.2f close=%.2f csync=%.2f => write_total=%.2f ms | "
                "evict=%.2f open=%.2f read=%.2f => read_total=%.2f ms | "
                "ratio=%.2fx ===\n",
                acc.n, acc.create_ms, acc.write_ms, acc.flush_ms, acc.sync_ms,
                acc.close_ms, acc.csync_ms, file_wtotal,
                acc.evict_ms, acc.open_ms, acc.read_ms, file_rtotal, file_ratio);

    bench_csv_row(csv, h5base, "ALL", "vol", "write", "file", file_wtotal, file_ratio, -1.0);
    bench_csv_row(csv, h5base, "ALL", "vol", "read",  "file", file_rtotal, -1.0,       -1.0);

    std::fclose(csv);
    std::fclose(xcsv);
    std::fprintf(stderr, "[dbg main] done: %d datasets, %d skipped, "
                 "%d measurements\n", n_ok, n_skip, acc.n);
    return 0;
}