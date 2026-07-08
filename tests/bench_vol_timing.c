#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define BENCH_DATASETS_ENABLE_HDF5
#include "bench_datasets.h"
#include "bench_compressors.h"
#include "bench_timing.h"

void  register_vol_properties(void);
hid_t make_dcpl(const char *compressor, const char *opts_json);

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
    s.rmse = sqrt(se / (double)nelem);
    return s;
}

static void *load_field(const bench_dataset_t *d, size_t *nbytes_out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = malloc(nbytes);
    if (!buf) { fprintf(stderr, "OOM %s\n", d->name); return NULL; }
    FILE *f = fopen(d->path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", d->path); free(buf); return NULL; }
    size_t got = fread(buf, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) {
        fprintf(stderr, "%s: read %zu/%zu bytes\n", d->name, got, nbytes);
        free(buf); return NULL;
    }
    *nbytes_out = nbytes;
    return buf;
}

static int name_selected(const char *sel, const char *name) {
    if (!sel || !*sel) return 1;
    size_t nl = strlen(name);
    for (const char *p = strstr(sel, name); p; p = strstr(p + 1, name)) {
        char b = (p == sel) ? ',' : p[-1], a = p[nl];
        if ((b == ',' || b == ' ') && (a == ',' || a == ' ' || a == '\0')) return 1;
    }
    return 0;
}

static void csv_header(FILE *f) {
    fprintf(f, "dataset,compressor,op,total_ms,rmse,raw_bytes,storage_bytes,ratio\n");
}
static void csv_row(FILE *f, const char *dset, const char *comp, const char *op,
                    double total_ms, double rmse, size_t raw, hsize_t storage) {
    double ratio = (storage > 0) ? (double)raw / (double)storage : 0.0;
    fprintf(f, "%s,%s,%s,%.4f,", dset, comp, op, total_ms);
    if (rmse >= 0.0) fprintf(f, "%.6e,", rmse); else fprintf(f, ",");
    fprintf(f, "%zu,%llu,%.3f\n", raw, (unsigned long long)storage, ratio);
    fflush(f);
}

static void run_pair(hid_t file, const bench_dataset_t *d,
                     const bench_compressor_t *c, const void *hbuf,
                     void *rbuf, size_t raw_bytes, FILE *csv) {
    size_t nelem  = bench_num_elements(d);
    hid_t  ntype  = bench_dataset_h5native(d);
    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);

    char dsname[192];
    snprintf(dsname, sizeof(dsname), "%s_%s", d->name, c->name);

    /* ---- WRITE (compressor chosen here via dcpl properties) ---- */
    hid_t dcpl = make_dcpl(c->pressio_id, c->opts_json);   /* NULL,NULL => default */
    hid_t dset = H5Dcreate2(file, dsname, ntype, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) { fprintf(stderr, "H5Dcreate2 failed %s\n", dsname);
                    H5Pclose(dcpl); H5Sclose(space); return; }

    double w0 = bench_now_ms();
    herr_t wret = H5Dwrite(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    double wms = bench_now_ms() - w0;

    hsize_t storage = H5Dget_storage_size(dset);   /* on-disk size for ratio/E4 */
    H5Dclose(dset); H5Pclose(dcpl);
    if (wret < 0) { fprintf(stderr, "H5Dwrite failed %s\n", dsname); H5Sclose(space); return; }
    csv_row(csv, d->name, c->name, "write", wms, -1.0, raw_bytes, storage);

    /* ---- READ + fidelity ---- */
    memset(rbuf, 0, raw_bytes);
    dset = H5Dopen2(file, dsname, H5P_DEFAULT);
    double r0 = bench_now_ms();
    herr_t rret = H5Dread(dset, ntype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    double rms = bench_now_ms() - r0;
    H5Dclose(dset); H5Sclose(space);
    if (rret < 0) { fprintf(stderr, "H5Dread failed %s\n", dsname); return; }

    bench_stats st = bench_compute_stats(hbuf, rbuf, nelem, d->dtype);
    csv_row(csv, d->name, c->name, "read", rms, st.rmse, raw_bytes, 0);

    printf("  %-28s W=%8.2f ms  R=%8.2f ms  ratio=%6.2fx  RMSE=%.3e\n",
           dsname, wms, rms,
           storage ? (double)raw_bytes / (double)storage : 0.0, st.rmse);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *h5path  = (argc > 1) ? argv[1] : "bench_out.h5";
    const char *csvpath = (argc > 2) ? argv[2] : "results_vol.csv";
    const char *only     = getenv("BENCH_ONLY");   /* dataset filter    */
    const char *only_cmp = getenv("BENCH_COMP");   /* compressor filter */

    register_vol_properties();
    bench_datasets_validate();

    FILE *csv = fopen(csvpath, "w");
    if (!csv) { perror("csv"); return 1; }
    char cwd[4096];
    if (csvpath[0] == '/' || !getcwd(cwd, sizeof(cwd)))
        fprintf(stderr, "csv path: %s\n", csvpath);
    else
        fprintf(stderr, "csv path: %s/%s\n", cwd, csvpath);
    csv_header(csv);

    hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed\n"); fclose(csv); return 1; }

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (!name_selected(only, d->name)) continue;
        if (access(d->path, R_OK) != 0) { fprintf(stderr, "skip %s (path)\n", d->name); continue; }

        size_t raw = 0;
        void *hbuf = load_field(d, &raw);
        if (!hbuf) continue;
        void *rbuf = malloc(raw);
        if (!rbuf) { free(hbuf); continue; }

        printf("\n=== %s (%.1f MiB, %s) ===\n", d->name,
               raw / (1024.0 * 1024.0), bench_dtype_name(d->dtype));
        for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
            const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
            if (!name_selected(only_cmp, c->name)) continue;
            run_pair(file, d, c, hbuf, rbuf, raw, csv);
        }
        free(rbuf); free(hbuf);
    }

    H5Fclose(file);
    fclose(csv);
    fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}