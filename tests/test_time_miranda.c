#include "miranda.h"

typedef struct {
    const char *suffix;
    const char *compressor;
    const char *opts_json;
} run_config_t;

static const run_config_t RUNS[] = {
    {
        "noop",
        NULL,
        NULL
    },
    {
        "Pf48_cuszp",
        "cuszp",
        "{\"pressio:abs\": 1e-3, \"cuszp:mode_str\": \"outlier\"}"
    },
    {
        "sz3_1e3",
        "sz3",
        "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}"
    },
    {
        "sz3_1e6",
        "sz3",
        "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6}"
    },
    {
        "bzip2",
        "bzip2",
        "{\"bzip2:block_size\":9}"
    },
};
#define N_RUNS  (int)(sizeof(RUNS) / sizeof(RUNS[0]))

static int test_field(hid_t file_id,
                      const char *label,
                      const double *field,
                      size_t nelem,
                      int ndims,
                      const hsize_t *dims)
{
    struct timespec t0, t1;

    printf("\n--- Field: %s  (%zu doubles, %.1f MB) ---\n",
           label, nelem, (double)(nelem * sizeof(double)) / (1024.0 * 1024.0));
    fflush(stdout);

    hid_t space_id = H5Screate_simple(ndims, dims, NULL);

    double *rbuf = malloc(nelem * sizeof(double));
    if (!rbuf) { fprintf(stderr, "OOM rbuf\n"); H5Sclose(space_id); return -1; }

    for (int r = 0; r < N_RUNS; r++) {
        const run_config_t *rc = &RUNS[r];
        char dsname[256];
        snprintf(dsname, sizeof(dsname), "%s_%s", label, rc->suffix);

        /* Write */
        hid_t dcpl = make_dcpl(rc->compressor, rc->opts_json);
        hid_t dset = H5Dcreate2(file_id, dsname, H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "  H5Dcreate2 failed for %s\n", dsname);
            H5Pclose(dcpl); continue;
        }

        clock_gettime(CLOCK_MONOTONIC, &t0);
        herr_t wret = H5Dwrite(dset, H5T_NATIVE_DOUBLE,
                                H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        H5Dclose(dset);
        H5Pclose(dcpl);

        printf("  Write %-28s  compressor=%-8s  ret=%2d  time=%.3f ms\n",
               dsname,
               rc->compressor ? rc->compressor : DEFAULT_COMPRESSOR,
               (int)wret, elapsed_ms(t0, t1));
        fflush(stdout);

        if (wret < 0) { fprintf(stderr, "  H5Dwrite failed for %s\n", dsname); continue; }

        /* Read */
        memset(rbuf, 0, nelem * sizeof(double));
        dset = H5Dopen2(file_id, dsname, H5P_DEFAULT);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        herr_t rret = H5Dread(dset, H5T_NATIVE_DOUBLE,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        H5Dclose(dset);

        if (rret < 0) { fprintf(stderr, "  H5Dread failed for %s\n", dsname); continue; }

        Stats st = compute_stats(field, rbuf, nelem);
        printf("  Read  %-28s  ret=%2d  time=%.3f ms  "
               "range=[%10.4g, %10.4g]  mean=%10.4g  RMSE=%.4e\n",
               dsname, (int)rret, elapsed_ms(t0, t1),
               st.min, st.max, st.mean, st.rmse);

        /* Check */
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

int main(void)
{
    printf("SDRBench Miranda compression timing test starting\n");
    printf("Path: %s\n", MIRANDA_PATH);

    const char *compressor = getenv("HDF5_VOL_PRESSIO_COMPRESSOR");
    const char *level      = getenv("HDF5_VOL_PRESSIO_LEVEL");
    printf("HDF5_VOL_PRESSIO_COMPRESSOR = %s\n",
           compressor ? compressor : "(not set, defaulting to noop)");
    printf("HDF5_VOL_PRESSIO_LEVEL      = %s\n",
           level      ? level      : "(not set, defaulting to 1)");
    fflush(stdout);

    register_vol_properties();

    hid_t file_id = H5Fcreate("sdrbench_miranda_compression.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    const char *fields[] = {
        "density.d64",
        "diffusivity.d64",
        "pressure.d64",
        "velocityx.d64",
        "velocityy.d64",
        "velocityz.d64",
        "viscocity.d64",
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
            fprintf(stderr, "WARNING: %s has %zu elements, expected %zu\n",
                    fields[fi], nelem, MIR_NELEM);
        }

        any = 1;

        char label[128];
        strncpy(label, fields[fi], sizeof(label) - 1);
        char *dot = strrchr(label, '.');
        if (dot) *dot = '\0';

        rc |= test_field(file_id, label, data, nelem, 3, dims);
        free(data);
    }

    H5Fclose(file_id);

    if (!any) {
        fprintf(stderr, "\nNo fields loaded. Check path and filenames with:\n"
                        "  ls %s\n", MIRANDA_PATH);
        return 1;
    }

    printf("\nSDRBench Miranda compression timing test %s.\n",
           rc == 0 ? "PASSED" : "FINISHED WITH ERRORS");
    return rc;
}