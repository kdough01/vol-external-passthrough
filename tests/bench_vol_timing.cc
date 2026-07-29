/* ============================================================================
 * bench_vol_timing.cc  --  APPROACH 2 of 3: passthrough VOL connector
 * ----------------------------------------------------------------------------
 * Drives the compressor THROUGH the VOL: H5Dcreate/H5Dwrite/H5Dread with the
 * connector active. At the HARNESS level you can only measure TOTAL (H5Dwrite /
 * H5Dread wall time via steady_clock) -- compression and I/O happen inside the
 * connector. To get the compress-vs-io SPLIT, instrument compress.cc directly
 * (see the snippet in the chat / timer_placement notes): bracket the
 * pressio_compressor_compress call and the under-VOL write, emitting rows with
 * path="vol". Those connector-side rows share this CSV schema and merge in.
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

typedef struct { double min, max, mean, rmse; } bench_stats;

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
    s.mean = sum / (double)nelem;
    s.rmse = std::sqrt(se / (double)nelem);
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

static int bench_check_chunks(const void *orig, const void *dec, size_t nelem,
                              bench_dtype_t dt, int nchunks, double bound,
                              const char *label) {
    if (nchunks < 1) nchunks = 1;
    size_t base = nelem / (size_t)nchunks, rem = nelem % (size_t)nchunks, off = 0;
    int bad = 0;
    for (int c = 0; c < nchunks; ++c) {
        size_t cn = base + ((size_t)c < rem ? 1 : 0);   // if your VOL puts the whole
        // remainder in the LAST chunk, use: cn = base + (c==nchunks-1 ? rem : 0);
        double se = 0.0, maxae = 0.0; size_t nz = 0;
        for (size_t i = off; i < off + cn; ++i) {
            double o = (dt==BENCH_F64)? ((const double*)orig)[i] : (double)((const float*)orig)[i];
            double v = (dt==BENCH_F64)? ((const double*)dec )[i] : (double)((const float*)dec )[i];
            double e = o - v, ae = std::fabs(e);
            se += e*e; if (ae > maxae) maxae = ae;
            if (v == 0.0 && std::fabs(o) > bound) ++nz;
        }
        int chunk_bad = (maxae > 2.0 * bound);
        bad += chunk_bad;
        std::fprintf(stderr,
            "[chunkchk] %-28s chunk %2d/%-2d n=%zu rmse=%.3e maxae=%.3e fillzero=%zu %s\n",
            label, c, nchunks, cn, std::sqrt(se/cn), maxae, nz, chunk_bad ? "<-- BAD" : "");
        off += cn;
    }
    if (bad) std::fprintf(stderr, "[chunkchk] %-28s **%d/%d chunks bad**\n", label, bad, nchunks);
    return bad;
}

/* Locate contiguous corrupt blocks and map each to chunk + offset.
 * Targets GROSS corruption (error >> bound), so it ignores the near-bound
 * points that are legitimately quantized. off_in_chunk is the position inside
 * that chunk's decompressed output -> trace it into the payload reader. */
static int bench_locate_bad(const void *orig, const void *dec, size_t nelem,
                            bench_dtype_t dt, int nchunks, double bound,
                            const char *label) {
    if (nchunks < 1) nchunks = 1;
    const size_t dsize = bench_dtype_size(dt);
    const size_t base  = nelem / (size_t)nchunks;   // equal split (matches your VOL log);
    const size_t rem   = nelem % (size_t)nchunks;   // if your VOL dumps the remainder in the
                                                    // LAST chunk instead, adjust chunk_of below
    const double thr   = 2.0 * bound;               // healthy points sit at ~1.0-1.001x bound
    const size_t GAP   = 256;                        // merge bad runs split by < GAP good elems

    auto chunk_of = [&](size_t i, size_t *cstart) -> int {
        size_t big = rem * (base + 1);              // first `rem` chunks hold base+1
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
              "byte_off=%zu | chunk %d (starts elem %zu) off_in_chunk=%zu elem / %zu B | "
              "len=2^%.1f off_in_chunk=2^%.1f\n",
              label, nblocks, start, last_bad, len, nbad, blk_maxae,
              start * dsize, c, cstart, off, off * dsize,
              std::log2((double)len), std::log2((double)(off ? off : 1)));
    }
    std::fprintf(stderr, nblocks ? "[badblk] %-24s %d bad block(s)\n"
                                 : "[badblk] %-24s clean\n", label, nblocks);
    return nblocks;
}

static void *load_field(const bench_dataset_t *d, size_t *nbytes_out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = std::malloc(nbytes);
    if (!buf) { std::fprintf(stderr, "OOM %s\n", d->name); return NULL; }
    FILE *f = std::fopen(d->path, "rb");
    if (!f) { std::fprintf(stderr, "open %s failed\n", d->path); std::free(buf); return NULL; }
    size_t got = std::fread(buf, 1, nbytes, f);
    std::fclose(f);
    if (got != nbytes) {
        std::fprintf(stderr, "%s: read %zu/%zu bytes\n", d->name, got, nbytes);
        std::free(buf); return NULL;
    }
    *nbytes_out = nbytes;
    return buf;
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

typedef struct {
    double             write_ms;   /* sum of per-dataset H5Dwrite wall time */
    double             read_ms;    /* sum of per-dataset H5Dread  wall time */
    size_t             raw_bytes;  /* sum of uncompressed input bytes       */
    unsigned long long storage;    /* sum of on-disk storage bytes          */
    int                n;          /* dataset writes counted                */
} bench_file_acc;

static void run_pair(hid_t file, const bench_dataset_t *d,
                     const bench_compressor_t *c, const void *hbuf,
                     void *rbuf, size_t raw_bytes, FILE *csv,
                     bench_file_acc *acc) {
    const bool dbg = std::getenv("BENCH_DEBUG") != NULL;
  try {
    size_t  nelem = bench_num_elements(d);
    hid_t   ntype = bench_dataset_h5native(d);
    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);

    char dsname[192];
    std::snprintf(dsname, sizeof(dsname), "%s_%s", d->name, c->name);

    if (dbg) std::fprintf(stderr, "[dbg run_pair] BEGIN %-24s rank=%d nelem=%zu xform=%s\n",
                          dsname, d->rank, nelem, bench_xform_name(d->xform));

    char opts[256];
    bench_compressor_opts_json(c, d, opts, sizeof(opts));
    const char *oj = (std::strcmp(opts, "{}") == 0) ? NULL : opts;
    if (dbg) std::fprintf(stderr, "[dbg run_pair] %-24s pressio_id=%s opts=%s\n",
                          dsname, c->pressio_id ? c->pressio_id : "(null)", oj ? oj : "(default)");

    /* ---- WRITE ---- */
    hid_t dcpl = make_dcpl(c->pressio_id, oj);
    hid_t dset = H5Dcreate2(file, dsname, ntype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        std::fprintf(stderr, "[ERR] H5Dcreate2 failed %s\n", dsname);
        H5Pclose(dcpl); H5Sclose(space); return;
    }

    BenchCpuTimer wt; wt.start();
    herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    double wms = wt.stop_ms();

    hsize_t storage = H5Dget_storage_size(dset);
    H5Dclose(dset); H5Pclose(dcpl);
    if (wret < 0) {
        std::fprintf(stderr, "[ERR] H5Dwrite failed %s\n", dsname);
        H5Sclose(space); return;
    }
    double wratio = storage ? (double)raw_bytes / (double)storage : 0.0;
    bench_csv_row(csv, d->name, c->name, "vol", "write", "total", wms, wratio, -1.0);

    /* accumulate file-level totals (write side) */
    if (acc) { acc->write_ms += wms; acc->raw_bytes += raw_bytes;
               acc->storage += (unsigned long long)storage; }

    /* ---- READ + fidelity ---- */
    std::memset(rbuf, 0, raw_bytes);
    dset = H5Dopen2(file, dsname, H5P_DEFAULT);
    if (dset < 0) { std::fprintf(stderr, "[ERR] H5Dopen2 failed %s\n", dsname); H5Sclose(space); return; }

    BenchCpuTimer rt; rt.start();
    herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    double rms = rt.stop_ms();

    H5Dclose(dset); H5Sclose(space);
    if (rret < 0) { std::fprintf(stderr, "[ERR] H5Dread failed %s\n", dsname); return; }

    int nchunks = 1;
    if (const char *s = std::getenv("VOL_COMP_CHUNK_N")) nchunks = std::atoi(s);
    bench_zero_run(hbuf, rbuf, nelem, d->dtype, dsname);
    bench_check_chunks(hbuf, rbuf, nelem, d->dtype, nchunks, d->bound, dsname);
    bench_locate_bad(hbuf, rbuf, nelem, d->dtype, nchunks, d->bound, dsname);

    if (acc) { acc->read_ms += rms; acc->n += 1; }   /* count dataset once */

    bench_stats st_tx = bench_compute_stats(hbuf, rbuf, nelem, d->dtype, BENCH_XFORM_NONE);

    if (dbg) {
        bench_stats st_lin = (d->xform == BENCH_XFORM_NONE)
                           ? st_tx
                           : bench_compute_stats(hbuf, rbuf, nelem, d->dtype, d->xform);
        double range_tx = st_tx.max - st_tx.min;
        double arel_tx  = (range_tx > 0.0) ? st_tx.rmse / range_tx : 0.0;
        std::fprintf(stderr,
            "[dbg run_pair] %-24s rms=%.2f  RMSE_codec=%.6e  RMSE_physical=%.6e "
            "achieved_rel=%.3e (nominal=%.3e)\n",
            dsname, rms, st_tx.rmse, st_lin.rmse, arel_tx, d->bound);
        if (d->bound_mode == BENCH_BOUND_REL && arel_tx > 2.0 * d->bound)
            std::fprintf(stderr, "[dbg WARN] %-24s codec not honoring bound "
                         "(achieved_rel %.3e > nominal %.3e)\n",
                         dsname, arel_tx, d->bound);
    }

    bench_csv_row(csv, d->name, c->name, "vol", "read", "total", rms, -1.0, st_tx.rmse);

    std::printf("  %-28s W=%8.2f ms  R=%8.2f ms  ratio=%6.2fx  RMSE=%.3e%s\n",
                dsname, wms, rms, wratio, st_tx.rmse,
                (d->xform != BENCH_XFORM_NONE) ? " (log-space)" : "");
    std::fflush(stdout);

  } catch (const std::exception &e) {
      std::fprintf(stderr, "[ERR] run_pair %s/%s threw: %s (skipped; sweep continues)\n",
                   d->name, c->name, e.what());
      std::fflush(stderr);
  } catch (...) {
      std::fprintf(stderr, "[ERR] run_pair %s/%s threw unknown exception (skipped)\n",
                   d->name, c->name);
      std::fflush(stderr);
  }
}

int main(int argc, char **argv) {
    const char *h5path   = (argc > 1) ? argv[1] : "bench_out.h5";
    const char *csvpath  = (argc > 2) ? argv[2] : "results_vol.csv";
    const char *only     = std::getenv("BENCH_ONLY");
    const char *only_cmp = std::getenv("BENCH_COMP");
    const bool  dbg      = std::getenv("BENCH_DEBUG") != NULL;

    std::fprintf(stderr, "[dbg main] h5=%s csv=%s only=%s comp=%s debug=%d\n",
                 h5path, csvpath, only ? only : "(all)", only_cmp ? only_cmp : "(all)", (int)dbg);

    register_vol_properties();
    bench_datasets_validate();

    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    /* ---- file-level: time creation ---- */
    BenchCpuTimer fct; fct.start();
    hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    double create_ms = fct.stop_ms();
    if (file < 0) { std::fprintf(stderr, "[ERR] H5Fcreate failed\n"); std::fclose(csv); return 1; }

    bench_file_acc acc = {0.0, 0.0, 0, 0ULL, 0};
    int n_ok = 0, n_skip = 0;

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (!name_selected(only, d->name)) continue;
        if (access(d->path, R_OK) != 0) {
            std::fprintf(stderr, "[dbg main] skip %s (unreadable: %s)\n", d->name, d->path);
            n_skip++; continue;
        }

        bench_dataset_t rz;
        size_t raw = 0;
        void *hbuf = bench_load_field(d, &rz, &raw);
        if (!hbuf) { std::fprintf(stderr, "[ERR] load failed %s\n", d->name); n_skip++; continue; }

        if (rz.xform != BENCH_XFORM_NONE) {   /* log/asinh preprocessing */
            if (dbg) std::fprintf(stderr, "[dbg main] applying %s to %s\n",
                                  bench_xform_name(rz.xform), rz.name);
            bench_apply_xform(hbuf, bench_num_elements(&rz), rz.dtype, rz.xform);
        }

        {   /* range scan (post-transform) */
            size_t ne = bench_num_elements(&rz);
            double mn = DBL_MAX, mx = -DBL_MAX;
            for (size_t i = 0; i < ne; ++i) {
                double v = (rz.dtype == BENCH_F64) ? ((const double*)hbuf)[i]
                                                   : (double)((const float*)hbuf)[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            std::fprintf(stderr, "RANGE %-14s min=%.6e max=%.6e range=%.6e xform=%s\n",
                         rz.name, mn, mx, mx - mn, bench_xform_name(rz.xform));
        }

        void *rbuf = std::malloc(raw);
        if (!rbuf) { std::fprintf(stderr, "[ERR] OOM rbuf %s\n", rz.name); std::free(hbuf); continue; }

        std::printf("\n=== %s (%.1f MiB, %s, %s%s) ===\n", rz.name,
                    raw / (1024.0 * 1024.0), bench_dtype_name(rz.dtype), bench_src_name(rz.src),
                    rz.xform != BENCH_XFORM_NONE ? ", transformed" : "");

        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (!name_selected(only_cmp, c->name)) continue;
            run_pair(file, &rz, c, hbuf, rbuf, raw, csv, &acc);
        }
        std::free(rbuf); std::free(hbuf);
        n_ok++;
    }

    /* ---- file-level: time close/flush, emit file rows ---- */
    BenchCpuTimer fclt; fclt.start();
    H5Fclose(file);
    double close_ms = fclt.stop_ms();

    double file_wtotal = create_ms + acc.write_ms + close_ms;   /* full build incl. flush */
    double file_rtotal = acc.read_ms;
    double file_ratio  = acc.storage ? (double)acc.raw_bytes / (double)acc.storage : 0.0;

    std::printf("\n=== FILE %s: %d dataset-writes | create=%.2f  write=%.2f  close/flush=%.2f "
                "=> write_total=%.2f ms | read_total=%.2f ms | ratio=%.2fx ===\n",
                h5path, acc.n, create_ms, acc.write_ms, close_ms,
                file_wtotal, file_rtotal, file_ratio);
    bench_csv_row(csv, h5path, "ALL", "vol", "write", "file", file_wtotal, file_ratio, -1.0);
    bench_csv_row(csv, h5path, "ALL", "vol", "read",  "file", file_rtotal, -1.0,       -1.0);

    std::fclose(csv);
    std::fprintf(stderr, "[dbg main] done: %d datasets, %d skipped\n", n_ok, n_skip);
    return 0;
}