/*
 * test_sdrbench.c
 *
 * Tests the vol-external-passthrough VOL connector against the Miranda
 * SDRBench dataset on LCRC. Files are raw binary float64 (double).
 *
 * Dataset path:
 *   /lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384/
 *
 * Fields (all 256x384x384 float64):
 *   density.d64, diffusivity.d64, pressure.d64,
 *   velocityx.d64, velocityy.d64, velocityz.d64, viscocity.d64
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hdf5.h"

/* ------------------------------------------------------------------ */
/*  Dataset configuration                                              */
/* ------------------------------------------------------------------ */

#define MIRANDA_PATH \
    "/lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384"

#define MIR_NX    256
#define MIR_NY    384
#define MIR_NZ    384
#define MIR_NELEM ((size_t)MIR_NX * MIR_NY * MIR_NZ)   /* 37,748,736 */

/* ------------------------------------------------------------------ */
/*  VOL property helpers                                               */
/* ------------------------------------------------------------------ */

static void register_vol_properties(void)
{
    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0) {
        char d[64] = "noop";
        H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0) {
        char d[4096] = "";
        H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
}

static hid_t make_dcpl(const char *compressor, const char *json_opts)
{
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    char buf[64] = "noop";
    if (compressor) strncpy(buf, compressor, sizeof(buf) - 1);
    H5Pset(dcpl, "pressio:compressor", buf);
    if (json_opts) {
        char jbuf[4096] = "";
        strncpy(jbuf, json_opts, sizeof(jbuf) - 1);
        H5Pset(dcpl, "vol:options_json", jbuf);
    }
    return dcpl;
}

/* ------------------------------------------------------------------ */
/*  I/O helper                                                         */
/* ------------------------------------------------------------------ */

static double *read_raw_double(const char *path, size_t *out_nelem)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    if (fsize <= 0 || fsize % sizeof(double) != 0) {
        fprintf(stderr, "Bad file size %ld for %s\n", fsize, path);
        fclose(f); return NULL;
    }

    size_t nelem = (size_t)fsize / sizeof(double);
    double *data = malloc(nelem * sizeof(double));
    if (!data) { fprintf(stderr, "OOM for %zu doubles\n", nelem); fclose(f); return NULL; }

    size_t got = fread(data, sizeof(double), nelem, f);
    fclose(f);
    if (got != nelem) {
        fprintf(stderr, "Short read: got %zu / %zu\n", got, nelem);
        free(data); return NULL;
    }

    *out_nelem = nelem;
    return data;
}

/* ------------------------------------------------------------------ */
/*  Statistics helper                                                  */
/* ------------------------------------------------------------------ */

typedef struct { double min, max, mean, rmse; } Stats;

static Stats compute_stats(const double *orig, const double *decomp, size_t n)
{
    Stats s = { orig[0], orig[0], 0.0, 0.0 };
    double sum = 0, sse = 0;
    for (size_t i = 0; i < n; i++) {
        if (orig[i] < s.min) s.min = orig[i];
        if (orig[i] > s.max) s.max = orig[i];
        sum += orig[i];
        double diff = orig[i] - decomp[i];
        sse += diff * diff;
    }
    s.mean = sum / (double)n;
    s.rmse = sqrt(sse / (double)n);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Per-field test                                                     */
/* ------------------------------------------------------------------ */

static int test_field(hid_t file_id,
                      const char *label,
                      const double *field,
                      size_t nelem,
                      int ndims,
                      const hsize_t *dims)
{
    printf("\n--- Field: %s  (%zu doubles, %.1f MB) ---\n",
           label, nelem, (double)(nelem * sizeof(double)) / (1024.0 * 1024.0));
    fflush(stdout);

    hid_t space_id = H5Screate_simple(ndims, dims, NULL);

    struct {
        const char *suffix;
        const char *compressor;
        const char *json;
    } runs[] = {
        { "noop",    NULL,   NULL },
        { "sz3_1e3", "sz3",
          "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}" },
        { "sz3_1e6", "sz3",
          "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6}" },
        { "zstd",    "zstd",
          "{\"zstd:clevel\":3}" },
    };
    int nruns = (int)(sizeof(runs) / sizeof(runs[0]));

    double *rbuf = malloc(nelem * sizeof(double));
    if (!rbuf) { fprintf(stderr, "OOM rbuf\n"); H5Sclose(space_id); return -1; }

    for (int r = 0; r < nruns; r++) {
        char dsname[256];
        snprintf(dsname, sizeof(dsname), "%s_%s", label, runs[r].suffix);

        hid_t dcpl = make_dcpl(runs[r].compressor, runs[r].json);
        hid_t dset = H5Dcreate2(file_id, dsname, H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "  H5Dcreate2 failed for %s\n", dsname);
            H5Pclose(dcpl); continue;
        }

        herr_t wret = H5Dwrite(dset, H5T_NATIVE_DOUBLE,
                                H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
        H5Dclose(dset);
        H5Pclose(dcpl);
        if (wret < 0) { fprintf(stderr, "  H5Dwrite failed for %s\n", dsname); continue; }

        /* Read back */
        memset(rbuf, 0, nelem * sizeof(double));
        dset = H5Dopen2(file_id, dsname, H5P_DEFAULT);
        herr_t rret = H5Dread(dset, H5T_NATIVE_DOUBLE,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        H5Dclose(dset);
        if (rret < 0) { fprintf(stderr, "  H5Dread failed for %s\n", dsname); continue; }

        Stats st = compute_stats(field, rbuf, nelem);
        printf("  %-22s  range=[%10.4g, %10.4g]  mean=%10.4g  RMSE=%.4e\n",
               dsname, st.min, st.max, st.mean, st.rmse);

        /* Spot-check first 5 values */
        printf("    orig:   ");
        for (int i = 0; i < 5; i++) printf("%12.6e ", field[i]);
        printf("\n    decomp: ");
        for (int i = 0; i < 5; i++) printf("%12.6e ", rbuf[i]);
        printf("\n");
        fflush(stdout);
    }

    free(rbuf);
    H5Sclose(space_id);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("SDRBench Miranda VOL test starting\n");
    printf("Path: %s\n", MIRANDA_PATH);
    fflush(stdout);

    register_vol_properties();

    hid_t file_id = H5Fcreate("sdrbench_miranda.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    const char *fields[] = {
        "density.d64",
        "diffusivity.d64",
        "pressure.d64",
        "velocityx.d64",
        "velocityy.d64",
        "velocityz.d64",
        "viscocity.d64",   /* note: SDRBench spells it this way */
        NULL
    };

    hsize_t dims[3] = { MIR_NX, MIR_NY, MIR_NZ };
    int any = 0, rc = 0;

    for (int fi = 0; fields[fi]; fi++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", MIRANDA_PATH, fields[fi]);

        size_t nelem = 0;
        double *data = read_raw_double(path, &nelem);
        if (!data) { fprintf(stderr, "Skipping %s\n", fields[fi]); continue; }

        if (nelem != MIR_NELEM) {
            fprintf(stderr,
                    "WARNING: %s has %zu elements, expected %zu\n",
                    fields[fi], nelem, MIR_NELEM);
        }

        any = 1;

        /* Strip .d64 extension for HDF5 dataset label */
        char label[128];
        strncpy(label, fields[fi], sizeof(label) - 1);
        char *dot = strrchr(label, '.');
        if (dot) *dot = '\0';

        rc |= test_field(file_id, label, data, nelem, 3, dims);
        free(data);
    }

    H5Fclose(file_id);

    if (!any) {
        fprintf(stderr,
                "\nNo fields loaded. Check path and filenames with:\n"
                "  ls %s\n", MIRANDA_PATH);
        return 1;
    }

    printf("\nSDRBench Miranda VOL test %s.\n", rc == 0 ? "PASSED" : "FINISHED WITH ERRORS");
    return rc;
}