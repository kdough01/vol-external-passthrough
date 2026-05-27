#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hdf5.h"

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

    /* Write through HDF5 (VOL connector intercepts this) */
    hid_t file_id = H5Fcreate("hurricane_test.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id = H5Screate_simple(3, dims, NULL);
    hid_t dset_id = H5Dcreate2(file_id, "pressure", H5T_NATIVE_FLOAT,
                                space_id, H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, buf);

    /* Read it back and verify */
    float *verify = malloc(nbytes);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, verify);

    /* Compute max error */
    double max_err = 0.0;
    for (size_t i = 0; i < nelem; i++) {
        double err = fabs((double)buf[i] - (double)verify[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);
    printf("Lossless: %s\n", max_err == 0.0 ? "YES" : "NO");

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
    free(buf);
    free(verify);

    return 0;
}