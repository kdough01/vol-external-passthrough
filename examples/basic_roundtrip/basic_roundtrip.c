#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"

#define H5FILE_NAME "basic_roundtrip.h5"
#define DATASET_NAME "temperature"
#define NX 16
#define NY 16
#define RANK 2
#define NELEMS (NX * NY)

typedef struct {
    const char *file_name;
    const char *dataset_name;
    const char *compressor;
    hsize_t dims[RANK];
} example_config_t;

static int check_status(herr_t status, const char *operation)
{
    if (status < 0) {
        fprintf(stderr, "%s failed\n", operation);
        return 0;
    }
    return 1;
}

static void fill_data(float *data, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        data[i] = 20.0f + 4.0f * sinf((float)i * 0.05f);
}

static double max_abs_error(const float *expected, const float *actual, size_t count)
{
    double result = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double error = fabs((double)expected[i] - (double)actual[i]);
        if (error > result)
            result = error;
    }
    return result;
}

int main(void)
{
    const example_config_t config = {
        H5FILE_NAME, DATASET_NAME, "bzip2", {NX, NY}
    };
    float write_data[NELEMS];
    float read_data[NELEMS];
    hid_t file_id = H5I_INVALID_HID;
    hid_t space_id = H5I_INVALID_HID;
    hid_t dcpl_id = H5I_INVALID_HID;
    hid_t dataset_id = H5I_INVALID_HID;
    hid_t dxpl_id = H5I_INVALID_HID;
    int result = EXIT_FAILURE;

    setenv("HDF5_VOL_PRESSIO_COMPRESSOR", config.compressor, 1);
    fill_data(write_data, NELEMS);
    memset(read_data, 0, sizeof(read_data));

    file_id = H5Fcreate(config.file_name, H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);
    space_id = H5Screate_simple(RANK, config.dims, NULL);
    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    dataset_id = H5Dcreate2(file_id, config.dataset_name, H5T_NATIVE_FLOAT,
                            space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (file_id < 0 || space_id < 0 || dcpl_id < 0 || dataset_id < 0)
        goto done;

    if (!check_status(H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL,
                               H5S_ALL, H5P_DEFAULT, write_data), "H5Dwrite"))
        goto done;
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Pclose(dcpl_id);
    H5Fclose(file_id);
    dataset_id = H5I_INVALID_HID;
    space_id = H5I_INVALID_HID;
    dcpl_id = H5I_INVALID_HID;
    file_id = H5I_INVALID_HID;

    file_id = H5Fopen(config.file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, config.dataset_name, H5P_DEFAULT);
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    if (file_id < 0 || dataset_id < 0 || dxpl_id < 0)
        goto done;

    if (!check_status(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL,
                              H5S_ALL, dxpl_id, read_data), "H5Dread"))
        goto done;

    printf("compressor=%s max_abs_error=%.6e\n",
           config.compressor, max_abs_error(write_data, read_data, NELEMS));
    result = EXIT_SUCCESS;

done:
    if (dxpl_id >= 0) H5Pclose(dxpl_id);
    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (dcpl_id >= 0) H5Pclose(dcpl_id);
    if (space_id >= 0) H5Sclose(space_id);
    if (file_id >= 0) H5Fclose(file_id);
    return result;
}
