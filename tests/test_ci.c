#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>

#define NX 32
#define NY 32
#define NZ 32
#define NELEMS (NX * NY * NZ)

static void fill_data(double *buf, size_t n)
{
    for (size_t i=0;i<n;i++) {
        buf[i] = sin((double)i * 0.0);
    }
}

static double max_abs_diff(const double *a, const double *b, size_t n)
{
    double mx = 0.0;
    for (size_t i=0;i<n;i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static int test_roundtrip(const char *compressor, double tol)
{
    int ret = 0;
    double *wbuf = NULL;
    double *rbuf = NULL;
    hid_t fid = H5I_INVALID_HID;
    hid_t sid = H5I_INVALID_HID;
    hid_t did = H5I_INVALID_HID;
    hsize_t dims[3] = {NX, NY, NZ};
    char fname[256];

    snprintf(fname, sizeof(fname), "/tmp/test_ci_%s.h5", compressor);

    wbuf = (double *) malloc(NELEMS * sizeof(double));
    rbuf = (double *)calloc(NELEMS, sizeof(double));

    if (!wbuf || !rbuf) {
        fprintf(stderr, "[%s] FAIL: malloc\n", compressor);
        ret = 1;
        goto done;
    }

    fill_data(wbuf, NELEMS);

    setenv("HDF5_VOL_PRESSIO_COMPRESSOR", compressor, 1);
    setenv("HDF5_VOL_PRESSIO_LEVEL", "1", 1);

    fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "[%s] FAIL: H5Fcreate\n", compressor);
        ret = 1;
        goto done;
    }

    sid = H5Screate_simple(3, dims, NULL);
    if (sid < 0) {
        fprintf(stderr, "[%s] FAIL: H5Screate_simple\n", compressor);
        ret = 1;
        goto done;
    }

    did = H5Dcreate2(fid, "data", H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (did < 0) {
        fprintf(stderr, "[%s] FAIL: H5Dcreate2\n", compressor);
        ret = 1;
        goto done;
    }

    if (H5Dwrite(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf) < 0) {
        fprintf(stderr, "[%s] FAIL: H5Dwrite\n", compressor);
        ret = 1;
        goto done;
    }

    H5Dclose(did); did = H5I_INVALID_HID;
    H5Dclose(sid); sid = H5I_INVALID_HID;
    H5Dclose(fid); fid = H5I_INVALID_HID;

    fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "[%s] FAIL: H5Fopen\n", compressor);
        ret = 1;
        goto done;
    }

    did = H5Dopen2(fid, "data", H5P_DEFAULT);
    if (did < 0) {
        fprintf(stderr, "[%s] FAIL: H5Dopen2\n", compressor);
        ret = 1;
        goto done;
    }

    double err = max_abs_diff(wbuf, rbuf, NELEMS);
    if (err > tol) {
        fprintf(stderr, "[%s] FAIL: max abs error %.6e > tol %.6e\n", compressor, err, tol);
        ret = 1;
        goto done;
    }
    print("[%s] PASS (max abs error: %.6e)\n", compressor, err);

done:
    if (did != H5I_INVALID_HID) H5Dclose(did);
    if (sid != H5I_INVALID_HID) H5Dclose(sid);
    if (fid != H5I_INVALID_HID) H5Dclose(fid);
    free(wbuf);
    free(rbuf);
    return ret;
}

static int test_bad_compressor(void)
{
    hid_t fid = H5I_INVALID_HID;
    hid_t sid = H5I_INVALID_HID;
    hid_t did = H5I_INVALID_HID;
    hsize_t dims[3] = {NX, NY, NZ};
    int ret = 0;

    setenv("HDF5_VOL_PRESSIO_COMPRESSOR", "does_not_exist", 1);

    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    fid = H5Fcreate("/tmp/test_ci_bad.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (fid < 0) {
        printf("[bad comrpessor] PASS (file create rejected unavailable compressor)\n");
        goto done;
    }

    sid = H5Screate_simple(3, dims, NULL);
    did = H5Dcreate2(fid, "data", H5T_NATIVE_DOUBLE, sid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did >= 0) {
        fprintf(stderr, "[bad_compressor] FAIL: H5Dcreate2 should have returned error for unknown compressor\n");
        H5Dclose(did);
        ret = 1;
    } else {
        printf("[bad_compressor] PASS (H5Dcreate2 correctly rejected unknown compressor)\n");
    }

done:
    H5set_auto(H5E_DEFAULT, (H5E_auto_t)H5Eprint2, stderr);
    if (sid != H5I_INVALID_HID) H5Sclose(sid);
    if (fid != H5I_INVALID_HID) H5Fclose(fid);
    return ret;
}

int main(void)
{
    int failures = 0;
    printf("----- VOL Compression Tests -----");

    failures += test_roundtrip("noop", 0.0);
    failures += test_roundtrip("bzip2", 0.0);
    failures += test_roundtrip("sz3", 1e-3);
    failures += test_roundtrip("zfp", 1e-3);

    failures += test_bad_compressor();

    setenv("HDF5_VOL_PRESSIO_COMPRESSOR", "noop", 1);

    printf("\n----- %s (%d failures) -----\n",
            failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
            failures,
            failures == 1 ? "" : "s");

    return failures > 0 ? 1 : 0;
}