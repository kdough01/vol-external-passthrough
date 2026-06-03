#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "hdf5.h"

static void register_vol_properties() {
    if (H5Pexist(H5P_DATASET_CREATE_DEFAULT, "pressio:compressor") <= 0) {
        char default_comp[64] = "zstd";
        H5Pregister2(H5P_DATASET_CREATE_DEFAULT, "pressio:compressor",
                     sizeof(default_comp), default_comp,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    if (H5Pexist(H5P_DATASET_CREATE_DEFAULT, "vol:options_json") <= 0) {
        char default_json[4096] = "";
        H5Pregister2(H5P_DATASET_CREATE_DEFAULT, "vol:options_json",
                     sizeof(default_json), default_json,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
}

int main() {
    /* Hurricane ISABEL: 100x500x500, single-precision float */
    hsize_t dims[3] = {100, 500, 500};
    size_t nelem = 100 * 500 * 500;
    size_t nbytes = nelem * sizeof(float);

    /* Read the raw binary file */
    float *buf = malloc(nbytes);
    FILE *fp = fopen("/lcrc/project/ECP-EZ/public/compression/Hurricane-ISABEL/cleaned-data/Pf48.bin", "rb");
    if (!fp) { fprintf(stderr, "Cannot open input file\n"); return 1; }
    size_t nread = fread(buf, sizeof(float), nelem, fp);
    fclose(fp);
    printf("Read %zu elements (%zu bytes)\n", nread, nread * sizeof(float));

    register_vol_properties();

    hid_t file_id = H5Fcreate("hurricane_test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id = H5Screate_simple(3, dims, NULL);

    /* Create DCPL with bzip2 compressor */
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset(dcpl_id, "pressio:compressor", "bzip2");
    H5Pset(dcpl_id, "vol:options_json", "{\"bzip2:compression_level\": 5}");

    hid_t dset_id = H5Dcreate2(file_id, "pressure", H5T_NATIVE_FLOAT, space_id,
                                H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset_id);
    H5Pclose(dcpl_id);
    H5Sclose(space_id);

    /* Reopen and read back */
    hid_t space_id2 = H5Screate_simple(3, dims, NULL);
    dset_id = H5Dopen2(file_id, "pressure", H5P_DEFAULT);
    float *verify = malloc(nbytes);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, verify);

    /* Compute max error */
    double max_err = 0.0;
    for (size_t i = 0; i < nelem; i++) {
        double err = fabs((double)buf[i] - (double)verify[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);
    printf("Lossless: %s\n", max_err == 0.0 ? "YES" : "NO");

    H5Dclose(dset_id);
    H5Sclose(space_id2);
    H5Fclose(file_id);
    free(buf);
    free(verify);

    return 0;
}