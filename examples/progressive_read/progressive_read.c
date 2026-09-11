#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"
#include "../../src/compress/vol_progressive.h"

#define H5FILE_NAME "progressive_read.h5"
#define DATASET_NAME "field"
#define NX 32
#define NY 32
#define NZ 32
#define RANK 3
#define NELEMS (NX * NY * NZ)

typedef struct {
    const char *compressor;
    const char *options_json;
    unsigned requested_percent;
} progressive_config_t;

static void fill_data(float *data)
{
    for (size_t i = 0; i < NELEMS; ++i)
        data[i] = sinf((float)i * 0.01f) + 0.25f * cosf((float)i * 0.003f);
}

static double rms_error(const float *expected, const float *actual)
{
    double sum = 0.0;
    for (size_t i = 0; i < NELEMS; ++i) {
        double error = (double)expected[i] - (double)actual[i];
        sum += error * error;
    }
    return sqrt(sum / (double)NELEMS);
}

static int register_vol_properties(void)
{
    char default_compressor[64] = "noop";
    char default_options[4096] = "";

    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0 &&
        H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                    sizeof(default_compressor), default_compressor,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL) < 0)
        return 0;
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0 &&
        H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                    sizeof(default_options), default_options,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL) < 0)
        return 0;
    return 1;
}

int main(void)
{
    const progressive_config_t config = {
        "sperr", "{\"vol:chunking_mode\": \"vol\", \"vol:chunk_n\": 4096}", 25
    };
    float write_data[NELEMS];
    float full_data[NELEMS];
    float reduced_data[NELEMS];
    hsize_t dims[RANK] = {NX, NY, NZ};
    hid_t file_id = H5I_INVALID_HID;
    hid_t space_id = H5I_INVALID_HID;
    hid_t dcpl_id = H5I_INVALID_HID;
    hid_t dataset_id = H5I_INVALID_HID;
    hid_t dxpl_id = H5I_INVALID_HID;
    int result = EXIT_FAILURE;

    if (!register_vol_properties())
        return EXIT_FAILURE;
    setenv("HDF5_VOL_PRESSIO_COMPRESSOR", config.compressor, 1);
    fill_data(write_data);
    memset(full_data, 0, sizeof(full_data));
    memset(reduced_data, 0, sizeof(reduced_data));

    file_id = H5Fcreate(H5FILE_NAME, H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);
    space_id = H5Screate_simple(RANK, dims, NULL);
    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset(dcpl_id, "pressio:compressor", config.compressor);
    H5Pset(dcpl_id, "vol:options_json", config.options_json);
    dataset_id = H5Dcreate2(file_id, DATASET_NAME, H5T_NATIVE_FLOAT,
                            space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if (file_id < 0 || space_id < 0 || dcpl_id < 0 || dataset_id < 0)
        goto done;
    if (H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, write_data) < 0)
        goto done;
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Pclose(dcpl_id);
    H5Fclose(file_id);
    dataset_id = H5I_INVALID_HID;
    space_id = H5I_INVALID_HID;
    dcpl_id = H5I_INVALID_HID;
    file_id = H5I_INVALID_HID;

    file_id = H5Fopen(H5FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, DATASET_NAME, H5P_DEFAULT);
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    if (file_id < 0 || dataset_id < 0 || dxpl_id < 0)
        goto done;
    if (H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, full_data) < 0)
        goto done;
    if (H5Pset_vol_progressive_pct(dxpl_id, config.requested_percent) < 0)
        goto done;
    if (H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                dxpl_id, reduced_data) < 0)
        goto done;

    printf("compressor=%s requested_percent=%u full_rms=%.6e reduced_rms=%.6e\n",
           config.compressor, config.requested_percent,
           rms_error(write_data, full_data), rms_error(write_data, reduced_data));
    result = EXIT_SUCCESS;

done:
    if (dxpl_id >= 0) H5Pclose(dxpl_id);
    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (dcpl_id >= 0) H5Pclose(dcpl_id);
    if (space_id >= 0) H5Sclose(space_id);
    if (file_id >= 0) H5Fclose(file_id);
    return result;
}
