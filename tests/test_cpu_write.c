#include <stdio.h>
#include <string.h>
#include "hdf5.h"

static void register_vol_properties() {
    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0) {
        char default_comp[64] = "noop";
        H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                     sizeof(default_comp), default_comp,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0) {
        char default_json[4096] = "";
        H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                     sizeof(default_json), default_json,
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL);
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

int main() {
    printf("Starting test\n"); fflush(stdout);
    register_vol_properties();

    hid_t file_id = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    printf("File created: %lld\n", (long long)file_id); fflush(stdout);

    hsize_t dims[2] = {5, 4};
    hid_t space_id = H5Screate_simple(2, dims, NULL);
    float buf[20];

    /* ---- bzip2 level 5 ---- */
    hid_t dcpl = make_dcpl("bzip2", "{\"bzip2:compression_level\": 5}");
    hid_t dset = H5Dcreate2(file_id, "bzip2_l5", H5T_NATIVE_FLOAT, space_id,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- bzip2 level 9 ---- */
    dcpl = make_dcpl("bzip2", "{\"bzip2:compression_level\": 9}");
    dset = H5Dcreate2(file_id, "bzip2_l9", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 1.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- sz3 absolute error bound 1e-3 ---- */
    dcpl = make_dcpl("sz3",
        "{\"sz3:error_bound_mode_str\": \"abs\", \"sz3:abs_error_bound\": 1e-3}");
    dset = H5Dcreate2(file_id, "sz3_abs1e3", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- sz3 absolute error bound 1e-6 (tighter) ---- */
    dcpl = make_dcpl("sz3",
        "{\"sz3:error_bound_mode_str\": \"abs\", \"sz3:abs_error_bound\": 1e-6}");
    dset = H5Dcreate2(file_id, "sz3_abs1e6", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- zfp fixed-accuracy 1e-3 ---- */
    dcpl = make_dcpl("zfp",
        "{\"zfp:type\": \"a\", \"zfp:accuracy\": 1e-3}");
    dset = H5Dcreate2(file_id, "zfp_acc1e3", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- zfp fixed-rate 8 bits/value ---- */
    dcpl = make_dcpl("zfp",
        "{\"zfp:type\": \"r\", \"zfp:rate\": 8.0}");
    dset = H5Dcreate2(file_id, "zfp_rate8", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 0.5f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    /* ---- default / noop ---- */
    dcpl = make_dcpl(NULL, NULL);
    dset = H5Dcreate2(file_id, "default_data", H5T_NATIVE_FLOAT, space_id,
                      H5P_DEFAULT, dcpl, H5P_DEFAULT);
    for (int i = 0; i < 20; i++) buf[i] = (float)i * 2.0f;
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset); H5Pclose(dcpl);

    H5Sclose(space_id);

    /* ---- read back all ---- */
    const char *names[] = {
        "bzip2_l5", "bzip2_l9",
        "sz3_abs1e3", "sz3_abs1e6",
        "zfp_acc1e3", "zfp_rate8",
        "default_data"
    };
    int n = sizeof(names) / sizeof(names[0]);
    for (int d = 0; d < n; d++) {
        memset(buf, 0, sizeof(buf));
        dset = H5Dopen2(file_id, names[d], H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        printf("%s: ", names[d]);
        for (int i = 0; i < 20; i++) printf("%.4f ", buf[i]);
        printf("\n");
        H5Dclose(dset);
    }

    H5Fclose(file_id);
    printf("Done.\n");
    return 0;
}