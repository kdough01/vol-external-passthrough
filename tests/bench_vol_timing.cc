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

#define BENCH_CONFIG_ENABLE_HDF5
#include "bench_config.h"
#include "bench_timing.h"
#include "miranda.h"     /* register_vol_properties(), make_dcpl(id, opts_json) */

typedef struct { double min, max, mean, rmse; } bench_stats;

static bench_stats bench_compute_stats(const void *orig, const void *dec,
                                       size_t nelem, bench_dtype_t dt) {
    double mn = DBL_MAX, mx = -DBL_MAX, sum = 0.0, se = 0.0;
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
    }
    bench_stats s;
    s.min = mn; s.max = mx;
    s.mean = sum / (double)nelem;
    s.rmse = std::sqrt(se / (double)nelem);
    return s;
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

static void run_pair(hid_t file, const bench_dataset_t *d,
                     const bench_compressor_t *c, const void *hbuf,
                     void *rbuf, size_t raw_bytes, FILE *csv) {
    size_t  nelem = bench_num_elements(d);
    hid_t   ntype = bench_dataset_h5native(d);
    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);

    char dsname[192];
    std::snprintf(dsname, sizeof(dsname), "%s_%s", d->name, c->name);

    /* ---- WRITE (compressor chosen here via dcpl properties) ---- */
    hid_t dcpl = make_dcpl(c->pressio_id, c->opts_json);   /* NULL,NULL => default */
    hid_t dset = H5Dcreate2(file, dsname, ntype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        std::fprintf(stderr, "H5Dcreate2 failed %s\n", dsname);
        H5Pclose(dcpl); H5Sclose(space); return;
    }

    BenchCpuTimer wt; wt.start();
    herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    double wms = wt.stop_ms();

    hsize_t storage = H5Dget_storage_size(dset);          /* on-disk size for ratio */
    H5Dclose(dset); H5Pclose(dcpl);
    if (wret < 0) {
        std::fprintf(stderr, "H5Dwrite failed %s\n", dsname);
        H5Sclose(space); return;
    }
    double wratio = storage ? (double)raw_bytes / (double)storage : 0.0;
    bench_csv_row(csv, d->name, c->name, "vol", "write", "total",
                  wms, wratio, -1.0);

    /* ---- READ + fidelity ---- */
    std::memset(rbuf, 0, raw_bytes);
    dset = H5Dopen2(file, dsname, H5P_DEFAULT);

    BenchCpuTimer rt; rt.start();
    herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    double rms = rt.stop_ms();

    H5Dclose(dset); H5Sclose(space);
    if (rret < 0) { std::fprintf(stderr, "H5Dread failed %s\n", dsname); return; }

    bench_stats st = bench_compute_stats(hbuf, rbuf, nelem, d->dtype);
    bench_csv_row(csv, d->name, c->name, "vol", "read", "total",
                  rms, -1.0, st.rmse);

    std::printf("  %-28s W=%8.2f ms  R=%8.2f ms  ratio=%6.2fx  RMSE=%.3e\n",
                dsname, wms, rms, wratio, st.rmse);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    const char *h5path   = (argc > 1) ? argv[1] : "bench_out.h5";
    const char *csvpath  = (argc > 2) ? argv[2] : "results_vol.csv";
    const char *only     = std::getenv("BENCH_ONLY");   /* dataset filter    */
    const char *only_cmp = std::getenv("BENCH_COMP");   /* compressor filter */

    register_vol_properties();
    bench_datasets_validate();

    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { std::fprintf(stderr, "H5Fcreate failed\n"); std::fclose(csv); return 1; }

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (!name_selected(only, d->name)) continue;
        if (access(d->path, R_OK) != 0) {
            std::fprintf(stderr, "skip %s (path)\n", d->name); continue;
        }
        size_t raw = 0;
        void *hbuf = load_field(d, &raw);
        if (!hbuf) continue;
        void *rbuf = std::malloc(raw);
        if (!rbuf) { std::free(hbuf); continue; }

        std::printf("\n=== %s (%.1f MiB, %s) ===\n", d->name,
                    raw / (1024.0 * 1024.0), bench_dtype_name(d->dtype));

        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (!name_selected(only_cmp, c->name)) continue;
            run_pair(file, d, c, hbuf, rbuf, raw, csv);
        }
        std::free(rbuf); std::free(hbuf);
    }

    H5Fclose(file);
    std::fclose(csv);
    std::fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}