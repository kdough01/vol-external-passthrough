#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hdf5.h"
#include "bench_config.h"

#define DEFAULT_COMPRESSOR "noop"

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

static hid_t dt_h5(bench_dtype_t t) {
    return (t == DT_F32) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
}

static void register_vol_properties(void) {
    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0) {
        static char d[64] = "noop";
        H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0) {
        static char d[4096] = "";
        H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
}

static hid_t make_dcpl(const char *compressor, const char *json_opts) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    char buf[64];
    strncpy(buf, compressor ? compressor : DEFAULT_COMPRESSOR, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    H5Pset(dcpl, "pressio:compressor", buf);
    if (json_opts) {
        char jbuf[4096] = "";
        strncpy(jbuf, json_opts, sizeof(jbuf) - 1);
        H5Pset(dcpl, "vol:options_json", jbuf);
    }
    return dcpl;
}

static void *read_raw(const char *path, size_t nelem, bench_dtype_t t) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return NULL; }
    size_t es = bench_dt_size(t);
    void *buf = malloc(nelem * es);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, es, nelem, f);
    fclose(f);
    if (got != nelem) {
        fprintf(stderr, "  short read %s: got %zu, expected %zu\n", path, got, nelem);
        free(buf); return NULL;
    }
    return buf;
}

int main(void) {
    register_vol_properties();

    const char *scratch = getenv("BENCH_SCRATCH");
    if (!scratch || !*scratch) scratch = ".";

    printf("# VOL-path timing (through vol-external-passthrough)\n");
    printf("# %-20s %-8s %11s %11s %7s %11s %11s %11s %12s\n",
           "dataset", "comp", "write_ms", "read_ms", "ratio",
           "min", "max", "mean", "rmse");
    fflush(stdout);

    for (int d = 0; d < BENCH_N_DATASETS; d++) {
        const bench_dataset_t *ds = &BENCH_DATASETS[d];
        size_t nelem = ds->nx * ds->ny * ds->nz;
        size_t esize = bench_dt_size(ds->dtype);

        void *field = read_raw(ds->path, nelem, ds->dtype);
        if (!field) { fprintf(stderr, "  skipping %s (read failed)\n", ds->name); continue; }

        char h5path[1024];
        snprintf(h5path, sizeof(h5path), "%s/bench_%s.h5", scratch, ds->name);
        hid_t fid = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0) { fprintf(stderr, "  H5Fcreate failed for %s\n", h5path); free(field); continue; }

        hsize_t dims[3] = { ds->nx, ds->ny, ds->nz };
        hid_t sid   = H5Screate_simple(3, dims, NULL);
        hid_t htype = dt_h5(ds->dtype);
        void *rbuf  = malloc(nelem * esize);

        for (int c = 0; c < BENCH_N_COMPRESSORS; c++) {
            const bench_comp_t *cc = &BENCH_COMPRESSORS[c];
            char dname[160];
            snprintf(dname, sizeof(dname), "%s_%s", ds->name, cc->label);

            hid_t dcpl = make_dcpl(cc->pressio_id, cc->opts_json);
            hid_t dset = H5Dcreate2(fid, dname, htype, sid,
                                    H5P_DEFAULT, dcpl, H5P_DEFAULT);
            if (dset < 0) {
                printf("  %-20s %-8s   (dataset create failed -- compressor unavailable?)\n",
                       ds->name, cc->label);
                H5Pclose(dcpl);
                fflush(stdout);
                continue;
            }

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            herr_t wret = H5Dwrite(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double wms = elapsed_ms(t0, t1);

            hsize_t storage = H5Dget_storage_size(dset);
            H5Dclose(dset);

            hid_t rdset = H5Dopen2(fid, dname, H5P_DEFAULT);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            herr_t rret = H5Dread(rdset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double rms = elapsed_ms(t0, t1);
            H5Dclose(rdset);

            if (wret < 0 || rret < 0) {
                printf("  %-20s %-8s   (io error wret=%d rret=%d)\n",
                       ds->name, cc->label, (int)wret, (int)rret);
                H5Pclose(dcpl);
                fflush(stdout);
                continue;
            }

            Stats st = bench_compute_stats(field, rbuf, nelem, ds->dtype);
            double ratio = (storage > 0)
                         ? (double)(nelem * esize) / (double)storage : 0.0;

            printf("  %-20s %-8s %11.2f %11.2f %7.2f %11.4g %11.4g %11.4g %12.4e\n",
                   ds->name, cc->label, wms, rms, ratio,
                   st.min, st.max, st.mean, st.rmse);
            fflush(stdout);

            H5Pclose(dcpl);
        }

        free(rbuf);
        H5Sclose(sid);
        H5Fclose(fid);
        free(field);
    }

    return 0;
}