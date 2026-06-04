#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"

#define HURRICANE_PATH "/lcrc/project/ECP-EZ/public/compression/Hurricane-ISABEL/cleaned-data"
#define NX 100
#define NY 500
#define NZ 500
#define NELEM (NX * NY * NZ)

static void register_vol_properties() {
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

static hid_t make_dcpl(const char *compressor, const char *json_opts) {
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

/* Read a raw binary float field from Hurricane-ISABEL */
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

int main() {
    printf("GPU Hurricane test starting\n"); fflush(stdout);
    register_vol_properties();

    /* Load Pf48 (pressure field) */
    char path[512];
    snprintf(path, sizeof(path), "%s/Pf48.bin", HURRICANE_PATH);
    float *field = read_hurricane_field(path);
    if (!field) return 1;
    printf("Loaded %s (%d floats)\n", path, NELEM); fflush(stdout);

    hid_t file_id = H5Fcreate("hurricane_gpu.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {NX, NY, NZ};
    hid_t space_id = H5Screate_simple(3, dims, NULL);

    /* --- GPU compressor (nvcomp_lz4 or whatever your CUDA build exposes) --- */
    /* Adjust the compressor name to match your gpu_compression registration  */
    hid_t dcpl = make_dcpl("nvcomp_lz4", NULL);
    hid_t dset = H5Dcreate2(file_id, "Pf48_gpu_lz4", H5T_NATIVE_FLOAT, space_id,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
    H5Dclose(dset); H5Pclose(dcpl);

    /* --- Also write a noop reference so we can diff --- */
    dcpl = make_dcpl(NULL, NULL);
    dset = H5Dcreate2(file_id, "Pf48_reference", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
    H5Dclose(dset); H5Pclose(dcpl);

    /* --- Read back and spot-check first 10 values --- */
    float *rbuf = malloc(NELEM * sizeof(float));
    dset = H5Dopen2(file_id, "Pf48_gpu_lz4", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    H5Dclose(dset);

    printf("Pf48_gpu_lz4 first 10 values:\n");
    for (int i = 0; i < 10; i++) printf("  [%d] orig=%.6f  decomp=%.6f  diff=%.2e\n",
                                         i, field[i], rbuf[i], field[i] - rbuf[i]);

    H5Sclose(space_id);
    H5Fclose(file_id);
    free(field);
    free(rbuf);
    printf("GPU Hurricane test done.\n");
    return 0;
}