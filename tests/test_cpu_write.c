#include <stdio.h>
#include <string.h>
#include "hdf5.h"

static void register_vol_properties() {
    printf("Checking pressio:compressor\n"); fflush(stdout);
    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0) {
        printf("Registering pressio:compressor\n"); fflush(stdout);
        char default_comp[64] = "noop";
        herr_t ret = H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                     sizeof(default_comp), default_comp,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        printf("Registration returned: %d\n", ret); fflush(stdout);
    }
    printf("Checking pressio:json\n"); fflush(stdout);
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0) {
        printf("Registering pressio:json\n"); fflush(stdout);
        char default_json[4096] = "";
        herr_t ret = H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                     sizeof(default_json), default_json,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        printf("Registration returned: %d\n", ret); fflush(stdout);
    }
}

static hid_t make_dcpl(const char *compressor, const char *json_opts) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    char buf[64] = "noop";
    if (compressor)
        strncpy(buf, compressor, sizeof(buf) - 1);
    H5Pset(dcpl, "pressio:compressor", buf);
    if (json_opts) {
        char jbuf[4096] = "";
        strncpy(jbuf, json_opts, sizeof(jbuf) - 1);
        H5Pset(dcpl, "vol:options_json", jbuf);
    }
    return dcpl;
}

int main() {
    printf("Starting test\n");
    fflush(stdout);
    
    register_vol_properties();
    printf("Properties registered\n");
    fflush(stdout);

    hid_t file_id = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    printf("File created: %lld\n", (long long)file_id);
    fflush(stdout);
    // ...
    hsize_t dims[2] = {5, 4};
    hid_t space_id = H5Screate_simple(2, dims, NULL);
    float buf[20];

    register_vol_properties();

    // Dataset 1: zstd, level 3
    hid_t dcpl1 = make_dcpl("bzip2", "{\"bzip2:compression_level\": 5}");
    hid_t dset1 = H5Dcreate2(file_id, "bzip2_data_l5", H5T_NATIVE_FLOAT, space_id,
                              H5P_DEFAULT, dcpl1, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset1, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset1);
    H5Pclose(dcpl1);

    // Dataset 2: sz3 with absolute error bound
    hid_t dcpl2 = make_dcpl("bzip2", "{\"bzip2:compression_level\": 9}");
    hid_t dset2 = H5Dcreate2(file_id, "bzip2_data_l9", H5T_NATIVE_FLOAT, space_id,
                              H5P_DEFAULT, dcpl2, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 1.5f;
    H5Dwrite(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset2);
    H5Pclose(dcpl2);

    // Dataset 3: default compressor, no JSON override
    hid_t dcpl3 = make_dcpl(NULL, NULL);
    hid_t dset3 = H5Dcreate2(file_id, "default_data", H5T_NATIVE_FLOAT, space_id,
                              H5P_DEFAULT, dcpl3, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 2.0f;
    H5Dwrite(dset3, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset3);
    H5Pclose(dcpl3);

    H5Sclose(space_id);

    // Read back and verify all three
    const char *names[] = {"bzip2_data_l5", "bzip2_data_l9", "default_data"};
    for (int d = 0; d < 3; d++) {
        memset(buf, 0, sizeof(buf));
        hid_t dset = H5Dopen2(file_id, names[d], H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        printf("%s: ", names[d]);
        for (int i = 0; i < 20; i++) printf("%.2f ", buf[i]);
        printf("\n");
        H5Dclose(dset);
    }

    H5Fclose(file_id);
    return 0;
}