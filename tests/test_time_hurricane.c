#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hdf5.h"
#include <libpressio/libpressio.h>


#define DEFAULT_COMPRESSOR "noop"
#define DEFAULT_LEVEL 1

typedef struct {
    const char *dset_name;
    const char *compressor;
    const char *opts_json;
} test_case_t;

static const test_case_t TEST_CASES[] = {
    {
        "Pf48_nvcomp",
        "nvcomp",
        NULL
    },
    {
        "Pf48_cusz",
        "cusz",
        "{\"pressio:abs\": 1e-3}"
    },
    {
        "Pf48_cuszp",
        "cuszp",
        "{\"pressio:abs\": 1e-3, \"cuszp:mode_str\": \"outlier\"}"
    },
    {
        "Pf48_reference",
        NULL,
        NULL
    },
};

#define N_TEST_CASES  (int)(sizeof(TEST_CASES) / sizeof(TEST_CASES[0]))

#define HURRICANE_PATH "/lcrc/project/ECP-EZ/public/compression/Hurricane-ISABEL/nonclean-data"
#define NX    100
#define NY    500
#define NZ    500
#define NELEM (NX * NY * NZ)

static double elapsed_ms(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec  - t0.tv_sec)  * 1000.0 +
           (t1.tv_nsec - t0.tv_nsec) / 1e6;
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

static float *read_hurricane_field(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }
    float *data = malloc(NELEM * sizeof(float));
    size_t got = fread(data, sizeof(float), NELEM, f);
    fclose(f);
    if (got != NELEM) {
        fprintf(stderr, "Short read: got %zu, expected %d\n", got, NELEM);
        free(data); return NULL;
    }
    return data;
}

int main(void) {
    struct timespec t0, t1;

    printf("GPU Hurricane test starting\n"); fflush(stdout);
    register_vol_properties();

    /* Load field */
    char path[512];
    snprintf(path, sizeof(path), "%s/CLOUDf01.bin", HURRICANE_PATH);
    float *field = read_hurricane_field(path);
    if (!field) return 1;
    printf("Loaded %s (%d floats)\n", path, NELEM);

    printf("Raw field first 10 values:\n");
    for (int i = 0; i < 10; i++) printf("  [%d] = %.6e\n", i, field[i]);
    fflush(stdout);

    hid_t file_id = H5Fcreate("hurricane_gpu.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {NX, NY, NZ};
    hid_t space_id = H5Screate_simple(3, dims, NULL);

    /* ── Writes ─────────────────────────────────────────────────── */
    printf("\n--- Writes ---\n");
    for (int i = 0; i < N_TEST_CASES; i++) {
        const test_case_t *tc = &TEST_CASES[i];
        hid_t dcpl = make_dcpl(tc->compressor, tc->opts_json);
        hid_t dset = H5Dcreate2(file_id, tc->dset_name, H5T_NATIVE_FLOAT,
                                 space_id, H5P_DEFAULT, dcpl, H5P_DEFAULT);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, field);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        printf("  Write %-20s  compressor=%-10s  ret=%2d  time=%.3f ms\n",
               tc->dset_name,
               tc->compressor ? tc->compressor : DEFAULT_COMPRESSOR,
               (int)ret, elapsed_ms(t0, t1));
        fflush(stdout);

        H5Dclose(dset);
        H5Pclose(dcpl);
    }

    /* ── Reads ──────────────────────────────────────────────────── */
    printf("\n--- Reads ---\n");
    float *rbuf = malloc(NELEM * sizeof(float));

    for (int i = 0; i < N_TEST_CASES; i++) {
        const test_case_t *tc = &TEST_CASES[i];
        hid_t dset = H5Dopen2(file_id, tc->dset_name, H5P_DEFAULT);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        herr_t ret = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, rbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        printf("  Read  %-20s  ret=%2d  time=%.3f ms\n",
               tc->dset_name, (int)ret, elapsed_ms(t0, t1));
        printf("  First 10 values [%s]:\n", tc->dset_name);
        for (int j = 0; j < 10; j++)
            printf("    [%d] orig=%.6e  decomp=%.6e  diff=%.2e\n",
                   j, field[j], rbuf[j], field[j] - rbuf[j]);
        fflush(stdout);

        H5Dclose(dset);
    }

    free(rbuf);
    H5Sclose(space_id);
    H5Fclose(file_id);
    free(field);
    printf("\nGPU Hurricane test done.\n");
    return 0;
}