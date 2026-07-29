/* ============================================================================
 * bench_vol_timing.cc  --  APPROACH 2 of 3: passthrough VOL connector
 * ----------------------------------------------------------------------------
 * Drives the compressor THROUGH the VOL: H5Dcreate/H5Dwrite/H5Dread with the
 * connector active. The phase split (stage / compress / container / io) comes
 * from the connector itself via vol_timing_phases.c -> $VOL_PHASES_CSV. This
 * harness measures the envelope: write, explicit flush, read, ratio, fidelity.
 *
 * WHAT CHANGED
 * ------------
 * 1. BENCH_REPS. Every measurement used to be a cold process: run_vol_group
 *    sets BENCH_ONLY and BENCH_COMP to one value each, so exactly one H5Dwrite
 *    happened per process and it absorbed CUDA context creation, cuSZp module
 *    load, the first (expensive) cudaMalloc, and libpressio plugin
 *    construction. All one-time, all charged to the single timed write, and all
 *    invariant to the error bound -- which is at least as good an explanation
 *    for 1e-3 and 1e-6 looking identical as any theory about H2D traffic.
 *    Run BENCH_REPS=4 and compare r0 against r1..rN to settle it.
 *
 * 2. BENCH_WARMUP (default on). A tiny dataset is written through the VOL with
 *    the same codec before the timed reps, so device/plugin initialisation
 *    happens outside every timed region. Done through the VOL rather than with
 *    cudaFree(0) because this TU is built by h5c++ without CUDA linkage.
 *
 * 3. Per-write flush. H5Dwrite returns once data is in HDF5's cache; the real
 *    disk traffic used to happen at H5Fclose, outside every timed region. Now
 *    H5Fflush is timed per write and reported as its own column.
 *
 * 4. BENCH_VERIFY (default off) gates the three O(nelem) diagnostic passes.
 *    On einspline37 (3.39e9 elements) they were five full passes over 13.5 GB
 *    on every run, inside a 900 s timeout.
 *
 * 5. Fidelity thresholds now come from bench_abs_threshold(), i.e. from the
 *    bound the codec was actually configured with. They used to threshold on
 *    d->bound (nominal relative 1e-3) while the entries configure pressio:abs,
 *    so cuszp_1e6 was checked at 2e-3 against a 1e-6 bound and passed
 *    vacuously.
 *
 * 6. BENCH_COLD_READ (default on) closes and reopens the file between write
 *    and read so H5Dread doesn't hit the HDF5 cache. NOTE: this does not clear
 *    the OS page cache, so reads remain page-cache-warm. State that in methods.
 *
 * Build:
 *   h5c++ -O2 -std=c++17 bench_vol_timing.cc -lpressio -o bench_vol_timing
 * ==========================================================================*/
#include <hdf5.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <exception>

#define BENCH_CONFIG_ENABLE_HDF5
#include "bench_config.h"
#include "bench_timing.h"
#include "miranda.h"     /* register_vol_properties(), make_dcpl(id, opts_json) */

/* ---------------------------------------------------------------------------
 * Extended CSV.
 *
 * Written alongside the legacy bench_csv_row output so existing merges keep
 * working. The legacy 9-field schema has no room for a repetition index or a
 * separate flush column, and both are load-bearing now.
 *
 * bench_pressio_timing.cc and bench_filter_timing.cc need this same header if
 * you want to merge across all three harnesses.
 * ------------------------------------------------------------------------ */
static const char *XCSV_HEADER =
    "dataset,compressor,codec_kind,chunk_n,rep,"
    "logical_bytes,stored_bytes,ratio,"
    "write_ms,flush_ms,read_ms,"
    "rmse,abs_thresh,maxae,bound_ok\n";

static void xcsv_row(FILE *fp, const char *dset, const bench_compressor_t *c,
                     int chunk_n, int rep,
                     unsigned long long logical, unsigned long long stored,
                     double write_ms, double flush_ms, double read_ms,
                     double rmse, double abs_thresh, double maxae) {
    if (!fp) return;
    const double ratio = stored ? (double)logical / (double)stored : 0.0;
    const int bound_ok = (abs_thresh > 0.0) ? (maxae <= 2.0 * abs_thresh)
                                            : (maxae == 0.0);
    std::fprintf(fp,
        "%s,%s,%s,%d,%d,%llu,%llu,%.4f,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e,%d\n",
        dset, c->name, bench_codec_kind_name(c->kind), chunk_n, rep,
        logical, stored, ratio, write_ms, flush_ms, read_ms,
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

/* thr is now an ABSOLUTE error threshold from bench_abs_threshold(), not the
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

/* Locate contiguous corrupt blocks and map each to chunk + offset.
 * Targets GROSS corruption (error >> thr), so it ignores the near-bound points
 * that are legitimately quantized. */
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
    int n = std::atoi(s);
    return n;
}
static int bench_reps(void)      { int n = env_int("BENCH_REPS", 1);  return n > 0 ? n : 1; }
static int bench_warmup(void)    { return env_int("BENCH_WARMUP", 1) != 0; }
static int bench_verify(void)    { return env_int("BENCH_VERIFY", 0) != 0; }
static int bench_cold_read(void) { return env_int("BENCH_COLD_READ", 1) != 0; }

static int name_selected(const char *sel, const char *name) {
    if (!sel || !*sel) return 1;
    size_t nl = std::strlen(name);
    for (const char *p = std::strstr(sel, name); p; p = std::strstr(p + 1, name)) {
        char b = (p == sel) ? ',' : p[-1], a = p[nl];
        if ((b == ',' || b == ' ') && (a == ',' || a == ' ' || a == '\0')) return 1;
    }
    return 0;
}

typedef struct {
    double             write_ms;   /* sum of timed H5Dwrite wall time  */
    double             flush_ms;   /* sum of timed H5Fflush wall time  */
    double             read_ms;    /* sum of timed H5Dread  wall time  */
    size_t             raw_bytes;
    unsigned long long storage;
    int                n;          /* dataset writes counted           */
} bench_file_acc;

/* --------------------------------------------------------------------------
 * Warmup: force device context creation, codec module load, and libpressio
 * plugin construction to happen OUTSIDE any timed region. A tiny dataset
 * through the VOL is enough, and needs no CUDA linkage in this TU.
 * ----------------------------------------------------------------------- */
static void warmup_codec(hid_t file, const bench_dataset_t *d,
                         const bench_compressor_t *c, const char *oj) {
    char    name[224];
    hsize_t wdims[1] = { 65536 };
    hid_t   ntype = bench_dataset_h5native(d);
    size_t  nb    = (size_t)wdims[0] * bench_dtype_size(d->dtype);
    void   *tmp   = std::calloc(1, nb);

    if (!tmp) return;
    std::snprintf(name, sizeof(name), "_warmup_%s", c->name);

  try {
    hid_t space = H5Screate_simple(1, wdims, NULL);
    hid_t dcpl  = make_dcpl(c->pressio_id, oj);
    hid_t dset  = H5Dcreate2(file, name, ntype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset >= 0) {
        (void)H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp);
        (void)H5Dread (dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp);
        H5Dclose(dset);
    }
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Ldelete(file, name, H5P_DEFAULT);   /* keep the container tidy */
  } catch (...) {
    /* A codec may reject a tiny input; that's fine, it still initialised. */
    std::fprintf(stderr, "[warmup] %s threw on the warmup dataset (ignored)\n",
                 c->name);
  }
    std::free(tmp);
}

/* --------------------------------------------------------------------------
 * One (dataset x compressor) pair, repeated BENCH_REPS times.
 * filep is in/out: cold reads close and reopen the file.
 * ----------------------------------------------------------------------- */
static void run_pair(hid_t *filep, const char *h5path,
                     const bench_dataset_t *d, const bench_compressor_t *c,
                     const void *hbuf, void *rbuf, size_t raw_bytes,
                     double measured_range,
                     FILE *csv, FILE *xcsv, bench_file_acc *acc) {
    const bool dbg       = std::getenv("BENCH_DEBUG") != NULL;
    const int  reps      = bench_reps();
    const int  verify    = bench_verify();
    const int  cold      = bench_cold_read();
    const int  chunk_n   = bench_effective_chunk_n(c);
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

    if (bench_warmup())
        warmup_codec(*filep, d, c, oj);

    for (int r = 0; r < reps; ++r) {
        char dsname[224];
        std::snprintf(dsname, sizeof(dsname), "%s_%s_r%d", d->name, c->name, r);

        /* ---------------- WRITE ---------------- */
        hid_t dcpl = make_dcpl(c->pressio_id, oj);
        hid_t dset = H5Dcreate2(*filep, dsname, ntype, space, H5P_DEFAULT,
                                dcpl, H5P_DEFAULT);
        if (dset < 0) {
            std::fprintf(stderr, "[ERR] H5Dcreate2 failed %s\n", dsname);
            H5Pclose(dcpl);
            continue;
        }

        BenchCpuTimer wt; wt.start();
        herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
        double wms = wt.stop_ms();

        /* The write above only reaches HDF5's cache. Time the flush separately
         * so io means io. This does change the I/O pattern relative to a single
         * close-time flush -- say so in methods. */
        BenchCpuTimer ft; ft.start();
        (void)H5Fflush(*filep, H5F_SCOPE_LOCAL);
        double fms = ft.stop_ms();

        hsize_t storage = H5Dget_storage_size(dset);
        H5Dclose(dset);
        H5Pclose(dcpl);

        if (wret < 0) {
            std::fprintf(stderr, "[ERR] H5Dwrite failed %s\n", dsname);
            continue;
        }

        double wratio = storage ? (double)raw_bytes / (double)storage : 0.0;
        bench_csv_row(csv, d->name, c->name, "vol", "write", "total", wms, wratio, -1.0);
        bench_csv_row(csv, d->name, c->name, "vol", "write", "flush", fms, -1.0, -1.0);

        if (acc) { acc->write_ms += wms; acc->flush_ms += fms;
                   acc->raw_bytes += raw_bytes;
                   acc->storage += (unsigned long long)storage; }

        /* ---------------- COLD READ SETUP ---------------- */
        if (cold) {
            H5Fclose(*filep);
            *filep = H5Fopen(h5path, H5F_ACC_RDWR, H5P_DEFAULT);
            if (*filep < 0) {
                std::fprintf(stderr, "[ERR] reopen failed for cold read: %s\n", h5path);
                return;
            }
        }

        /* ---------------- READ ---------------- */
        std::memset(rbuf, 0, raw_bytes);
        dset = H5Dopen2(*filep, dsname, H5P_DEFAULT);
        if (dset < 0) {
            std::fprintf(stderr, "[ERR] H5Dopen2 failed %s\n", dsname);
            continue;
        }

        BenchCpuTimer rt; rt.start();
        herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        double rms = rt.stop_ms();
        H5Dclose(dset);

        if (rret < 0) {
            std::fprintf(stderr, "[ERR] H5Dread failed %s\n", dsname);
            continue;
        }

        if (acc) { acc->read_ms += rms; acc->n += 1; }

        /* ---------------- FIDELITY ---------------- */
        bench_stats st = bench_compute_stats(hbuf, rbuf, nelem, d->dtype,
                                             BENCH_XFORM_NONE);

        if (verify) {
            bench_zero_run(hbuf, rbuf, nelem, d->dtype, dsname);
            bench_check_chunks(hbuf, rbuf, nelem, d->dtype, chunk_n, thr, dsname);
            bench_locate_bad(hbuf, rbuf, nelem, d->dtype, chunk_n, thr, dsname);
        }

        if (c->lossless && st.maxae != 0.0)
            std::fprintf(stderr, "[FAIL] %-28s declared lossless but maxae=%.6e\n",
                         dsname, st.maxae);
        else if (thr > 0.0 && st.maxae > 2.0 * thr)
            std::fprintf(stderr, "[FAIL] %-28s maxae=%.6e exceeds 2x configured "
                                 "bound %.6e\n", dsname, st.maxae, thr);

        bench_csv_row(csv, d->name, c->name, "vol", "read", "total", rms, -1.0, st.rmse);
        xcsv_row(xcsv, d->name, c, chunk_n, r,
                 (unsigned long long)raw_bytes, (unsigned long long)storage,
                 wms, fms, rms, st.rmse, thr, st.maxae);

        std::printf("  %-30s r%-2d W=%8.2f F=%8.2f R=%8.2f ms  ratio=%6.2fx  "
                    "RMSE=%.3e maxae=%.3e%s\n",
                    dsname, r, wms, fms, rms, wratio, st.rmse, st.maxae,
                    (d->xform != BENCH_XFORM_NONE) ? " (log-space)" : "");
        std::fflush(stdout);
    }

    if (reps > 1)
        std::printf("  ^ compare r0 against r1..r%d: a large gap means you were "
                    "measuring initialisation, not steady-state cost\n", reps - 1);

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
    const char *h5path   = (argc > 1) ? argv[1] : "bench_out.h5";
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
        "[dbg main] h5=%s csv=%s xcsv=%s only=%s comp=%s "
        "reps=%d warmup=%d verify=%d cold_read=%d debug=%d\n",
        h5path, csvpath, xcsvpath, only ? only : "(all)",
        only_cmp ? only_cmp : "(all)",
        bench_reps(), bench_warmup(), bench_verify(), bench_cold_read(), (int)dbg);

    register_vol_properties();
    bench_datasets_validate();

    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    FILE *xcsv = std::fopen(xcsvpath, "w");
    if (!xcsv) { std::perror("xcsv"); std::fclose(csv); return 1; }
    std::fputs(XCSV_HEADER, xcsv);

    /* ---- file-level: time creation ---- */
    BenchCpuTimer fct; fct.start();
    hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    double create_ms = fct.stop_ms();
    if (file < 0) {
        std::fprintf(stderr, "[ERR] H5Fcreate failed\n");
        std::fclose(csv); std::fclose(xcsv); return 1;
    }

    bench_file_acc acc = {0.0, 0.0, 0.0, 0, 0ULL, 0};
    int n_ok = 0, n_skip = 0;

    const size_t mem_avail = bench_mem_available_bytes();

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (!name_selected(only, d->name)) continue;
        if (access(d->path, R_OK) != 0) {
            std::fprintf(stderr, "[dbg main] skip %s (unreadable: %s)\n", d->name, d->path);
            n_skip++; continue;
        }

        /* Fail fast instead of getting OOM-killed mid-sweep. einspline37 needs
         * ~26 GiB for hbuf+rbuf alone, which no 8 GiB job can satisfy. */
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
        if (!hbuf) { std::fprintf(stderr, "[ERR] load failed %s\n", d->name); n_skip++; continue; }

        if (rz.xform != BENCH_XFORM_NONE) {
            if (dbg) std::fprintf(stderr, "[dbg main] applying %s to %s\n",
                                  bench_xform_name(rz.xform), rz.name);
            bench_apply_xform(hbuf, bench_num_elements(&rz), rz.dtype, rz.xform);
        }

        /* Range scan (post-transform). Feeds bench_abs_threshold() for any
         * relative-bounded entry, so verification uses a measured range rather
         * than the assumed_range placeholder. */
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
            std::fprintf(stderr, "RANGE %-14s min=%.6e max=%.6e range=%.6e xform=%s\n",
                         rz.name, mn, mx, measured_range, bench_xform_name(rz.xform));
        }

        void *rbuf = std::malloc(raw);
        if (!rbuf) { std::fprintf(stderr, "[ERR] OOM rbuf %s\n", rz.name); std::free(hbuf); continue; }

        std::printf("\n=== %s (%.1f MiB, %s, %s%s) ===\n", rz.name,
                    raw / (1024.0 * 1024.0), bench_dtype_name(rz.dtype),
                    bench_src_name(rz.src),
                    rz.xform != BENCH_XFORM_NONE ? ", transformed" : "");

        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (!name_selected(only_cmp, c->name)) continue;
            run_pair(&file, h5path, &rz, c, hbuf, rbuf, raw,
                     measured_range, csv, xcsv, &acc);
        }
        std::free(rbuf); std::free(hbuf);
        n_ok++;
    }

    /* ---- file-level: time close/flush, emit file rows ---- */
    BenchCpuTimer fclt; fclt.start();
    H5Fclose(file);
    double close_ms = fclt.stop_ms();

    /* Per-dataset flush is now timed inside run_pair, so close_ms should be
     * small. It is still an aggregate over everything in the file: warn if more
     * than one write landed here, because then create/close are not
     * attributable to any single (dataset x compressor) pair. */
    if (acc.n > 1)
        std::fprintf(stderr,
            "[WARN] %d dataset-writes in one file: create_ms and close_ms are "
            "aggregates and are NOT per-pair attributable. Drive one dataset x "
            "one compressor per process (BENCH_ONLY + BENCH_COMP) if you need "
            "per-pair file-level numbers.\n", acc.n);

    double file_wtotal = create_ms + acc.write_ms + acc.flush_ms + close_ms;
    double file_rtotal = acc.read_ms;
    double file_ratio  = acc.storage ? (double)acc.raw_bytes / (double)acc.storage : 0.0;

    std::printf("\n=== FILE %s: %d dataset-writes | create=%.2f  write=%.2f  "
                "flush=%.2f  close=%.2f => write_total=%.2f ms | "
                "read_total=%.2f ms | ratio=%.2fx ===\n",
                h5path, acc.n, create_ms, acc.write_ms, acc.flush_ms, close_ms,
                file_wtotal, file_rtotal, file_ratio);
    bench_csv_row(csv, h5path, "ALL", "vol", "write", "file", file_wtotal, file_ratio, -1.0);
    bench_csv_row(csv, h5path, "ALL", "vol", "read",  "file", file_rtotal, -1.0,       -1.0);

    std::fclose(csv);
    std::fclose(xcsv);
    std::fprintf(stderr, "[dbg main] done: %d datasets, %d skipped\n", n_ok, n_skip);
    return 0;
}