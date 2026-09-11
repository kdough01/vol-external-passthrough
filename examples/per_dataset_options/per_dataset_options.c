#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"

#define H5FILE_NAME "per_dataset_options.h5"
#define RANK 2
#define NX 8
#define NY 8
#define NELEMS (NX * NY)

typedef enum {
    DATASET_BZIP2_LEVEL_1 = 0,
    DATASET_BZIP2_LEVEL_9 = 1
} dataset_case_t;

typedef struct {
    const char *name;
    const char *compressor;
    const char *options_json;
} dataset_config_t;

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

static hid_t make_dcpl(const dataset_config_t *config)
{
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    char compressor[64] = "noop";
    char options_json[4096] = "";

    if (dcpl_id < 0)
        return H5I_INVALID_HID;
    strncpy(compressor, config->compressor, sizeof(compressor) - 1);
    strncpy(options_json, config->options_json, sizeof(options_json) - 1);
    if (H5Pset(dcpl_id, "pressio:compressor", compressor) < 0 ||
        H5Pset(dcpl_id, "vol:options_json", options_json) < 0) {
        H5Pclose(dcpl_id);
        return H5I_INVALID_HID;
    }
    return dcpl_id;
}

static int create_dataset(hid_t file_id, const dataset_config_t *config,
                          float *data)
{
    hsize_t dims[RANK] = {NX, NY};
    hid_t space_id = H5Screate_simple(RANK, dims, NULL);
    hid_t dcpl_id = make_dcpl(config);
    hid_t dataset_id = H5Dcreate2(file_id, config->name, H5T_NATIVE_FLOAT,
                                  space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    int result = dataset_id >= 0 &&
        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                 H5P_DEFAULT, data) >= 0;

    if (dataset_id >= 0) H5Dclose(dataset_id);
    if (dcpl_id >= 0) H5Pclose(dcpl_id);
    if (space_id >= 0) H5Sclose(space_id);
    return result;
}

int main(void)
{
    const dataset_config_t configs[] = {
        {"level_1", "bzip2", "{\"bzip2:compression_level\": 1}"},
        {"level_9", "bzip2", "{\"bzip2:compression_level\": 9}"}
    };
    float data[NELEMS];
    hid_t file_id;

    if (!register_vol_properties()) {
        fprintf(stderr, "could not register VOL dataset properties\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < NELEMS; ++i)
        data[i] = (float)i * 0.25f;

    file_id = H5Fcreate(H5FILE_NAME, H5F_ACC_TRUNC,
                        H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        return EXIT_FAILURE;

    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        if (!create_dataset(file_id, &configs[i], data)) {
            H5Fclose(file_id);
            return EXIT_FAILURE;
        }
        printf("created %s with %s and options %s\n",
               configs[i].name, configs[i].compressor,
               configs[i].options_json);
    }
    H5Fclose(file_id);
    return EXIT_SUCCESS;
}
