#include "miranda.h"

static void test_error_handling(hid_t file_id, const double *field, size_t nelem,
                                int ndims, const hsize_t *dims)
{
    printf("\n=== Error Handling Tests ===\n");
    fflush(stdout);

    hid_t space_id = H5Screate_simple(ndims, dims, NULL);

    /* ---- Test 1: Compressor does not exist ---- */
    printf("\n-- Test 1: Unavailable compressor ('zstd') --\n");
    {
        hid_t dcpl = make_dcpl("zstd", "{\"zstd:clevel\":3}");
        hid_t dset = H5Dcreate2(file_id, "err_bad_compressor",
                                 H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0)
            printf("  PASS: H5Dcreate2 correctly failed for unavailable compressor\n");
        else {
            printf("  FAIL: H5Dcreate2 should have failed but succeeded\n");
            H5Dclose(dset);
        }
        H5Pclose(dcpl);
    }

    /* ---- Test 2: Bad JSON ---- */
    printf("\n-- Test 2: Malformed JSON options --\n");
    {
        hid_t dcpl = make_dcpl("sz3", "{this is not valid json!!!}");
        hid_t dset = H5Dcreate2(file_id, "err_bad_json",
                                 H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0)
            printf("  PASS: H5Dcreate2 correctly failed for bad JSON\n");
        else {
            printf("  FAIL: H5Dcreate2 should have failed but succeeded\n");
            H5Dclose(dset);
        }
        H5Pclose(dcpl);
    }

    /* ---- Test 3: Invalid option key for compressor ---- */
    printf("\n-- Test 3: Invalid JSON option key for compressor --\n");
    {
        hid_t dcpl = make_dcpl("sz3", "{\"sz3:this_key_does_not_exist\":999}");
        hid_t dset = H5Dcreate2(file_id, "err_bad_option",
                                 H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0)
            printf("  PASS: H5Dcreate2 correctly failed for invalid option key\n");
        else {
            printf("  INFO: H5Dcreate2 succeeded — libpressio silently ignored unknown key\n");
            H5Dclose(dset);
        }
        H5Pclose(dcpl);
    }

    /* ---- Test 4: noop is always allowed (intentional passthrough) ---- */
    printf("\n-- Test 4: noop passthrough (should always succeed) --\n");
    {
        hid_t dcpl = make_dcpl(NULL, NULL);
        hid_t dset = H5Dcreate2(file_id, "err_noop_passthrough",
                                 H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset >= 0) {
            herr_t wret = H5Dwrite(dset, H5T_NATIVE_DOUBLE,
                                   H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
            printf(wret == 0 ? "  PASS: noop write succeeded as expected\n"
                             : "  FAIL: noop write should have succeeded\n");
            H5Dclose(dset);
        } else {
            printf("  FAIL: noop H5Dcreate2 should have succeeded\n");
        }
        H5Pclose(dcpl);
    }

    /* ---- Test 5: Successful sz3 round-trip (regression check) ---- */
    printf("\n-- Test 5: Successful sz3 round-trip (regression check) --\n");
    {
        hid_t dcpl = make_dcpl("sz3",
            "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}");
        hid_t dset = H5Dcreate2(file_id, "err_sz3_roundtrip",
                                 H5T_NATIVE_DOUBLE, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            printf("  FAIL: sz3 H5Dcreate2 failed unexpectedly\n");
            H5Pclose(dcpl);
        } else {
            herr_t wret = H5Dwrite(dset, H5T_NATIVE_DOUBLE,
                                   H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
            H5Dclose(dset);
            H5Pclose(dcpl);

            if (wret < 0) {
                printf("  FAIL: sz3 write failed unexpectedly\n");
            } else {
                double *rbuf = calloc(nelem, sizeof(double));
                dset = H5Dopen2(file_id, "err_sz3_roundtrip", H5P_DEFAULT);
                herr_t rret = H5Dread(dset, H5T_NATIVE_DOUBLE,
                                      H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
                H5Dclose(dset);
                if (rret < 0) {
                    printf("  FAIL: sz3 read failed unexpectedly\n");
                } else {
                    Stats st = compute_stats(field, rbuf, nelem);
                    if (st.rmse <= 1e-3)
                        printf("  PASS: sz3 round-trip RMSE=%.4e within 1e-3 bound\n",
                               st.rmse);
                    else
                        printf("  FAIL: sz3 round-trip RMSE=%.4e exceeds 1e-3 bound\n",
                               st.rmse);
                }
                free(rbuf);
            }
        }
    }

    H5Sclose(space_id);
    printf("\n=== Error Handling Tests Complete ===\n");
    fflush(stdout);
}

int main(void)
{
    printf("SDRBench Miranda error handling test starting\n");
    printf("Path: %s\n", MIRANDA_PATH);
    fflush(stdout);

    register_vol_properties();

    hid_t file_id = H5Fcreate("sdrbench_miranda_errors.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    /* Use density as the representative field for all error tests */
    char path[512];
    snprintf(path, sizeof(path), "%s/density.d64", MIRANDA_PATH);
    size_t nelem = 0;
    double *data = read_raw_double(path, &nelem);
    if (!data) { H5Fclose(file_id); return 1; }

    hsize_t dims[3] = { MIR_NX, MIR_NY, MIR_NZ };
    test_error_handling(file_id, data, nelem, 3, dims);

    free(data);
    H5Fclose(file_id);

    printf("\nSDRBench Miranda error handling test done.\n");
    return 0;
}