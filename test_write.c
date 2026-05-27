#include <stdio.h>
#include "hdf5.h"

int main() {
    hid_t file_id = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[2] = {5, 4};
    hid_t space_id = H5Screate_simple(2, dims, NULL);
    hid_t dset_id = H5Dcreate2(file_id, "data", H5T_NATIVE_FLOAT, space_id,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    float buf[20];
    for (int i = 0; i < 20; i++) { buf[i] = i * 0.1f; }
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

    H5Dclose(dset_id);
    H5Sclose(space_id);

    hid_t dset_id_open = H5Dopen2(file_id, "data", H5P_DEFAULT);
    H5Dread(dset_id_open, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

    for (int i = 0; i < 20; i++) {
        printf("%.2f ", buf[i]);
    }
    printf("\n");

    H5Dclose(dset_id_open);
    H5Fclose(file_id);
    return 0;
}