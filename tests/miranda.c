/*
 * test_sdrbench.c
 *
 * Tests the vol-external-passthrough VOL connector against SDRBench datasets
 * available on the LCRC cluster. Targets Miranda velocity fields and HACC
 * cosmological simulation data — both raw binary float32 files.
 *
 * Dataset paths (adjust if the cluster layout differs):
 *   Miranda:  /lcrc/project/ECP-EZ/public/compression/Miranda/
 *   HACC:     /lcrc/project/ECP-EZ/public/compression/hacc-data/
 *
 * Miranda fields (all 256x384x384 float32, ~150 MB each):
 *   velocityx.f32, velocityy.f32, velocityz.f32, pressure.f32, density.f32
 *   (also viscosityx.f32 if present)
 *
 * HACC fields (1073726487 floats = ~4 GB; use the smaller snapshot instead):
 *   Use hacc-data/ sub-snapshots; see HACC_PATH below.
 *
 * Build: same CMakeLists as the hurricane test — just add this file to tests/.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hdf5.h"

/* ------------------------------------------------------------------ */
/*  Dataset configurations                                              */
/* ------------------------------------------------------------------ */

/* Miranda: 256 x 384 x 384  float32                                    */
#define MIRANDA_PATH "/lcrc/project/ECP-EZ/public/compression/Miranda"
#define MIR_NX  256
#define MIR_NY  384
#define MIR_NZ  384
#define MIR_NELEM ((size_t)MIR_NX * MIR_NY * MIR_NZ)   /* 37,748,736 */

/* HACC small snapshot: 280953867 particles × 1 float field             *
 * The "small" subset lives at hacc-data/; actual dims vary by file.    *
 * We treat each file as a 1-D array and read it whole.                 *
 * Uncomment and adjust if you want HACC instead.                       */
/* #define USE_HACC */
#define HACC_PATH  "/lcrc/project/ECP-EZ/public/compression/hacc-data"
#define HACC_FILE  "m000.full.physics.0101"   /* adjust to what's there */

/* ------------------------------------------------------------------ */
/*  VOL property helpers (identical to the hurricane test)             */
/* ------------------------------------------------------------------ */

static void register_vol_properties(void)
{
    if (H5Pexist(H5P_DATASET_CREATE, "pressio:compressor") <= 0) {
        char d[64] = "noop";
        H5Pregister2(H5P_DATASET_CREATE, "pressio:compressor",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    if (H5Pexist(H5P_DATASET_CREATE, "vol:options_json") <= 0) {
        char d[4096] = "";
        H5Pregister2(H5P_DATASET_CREATE, "vol:options_json",
                     sizeof(d), d, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
}

static hid_t make_dcpl(const char *compressor, const char *json_opts)
{
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

/* ------------------------------------------------------------------ */
/*  I/O helpers                                                        */
/* ------------------------------------------------------------------ */

/* Read an entire raw float32 binary file.
 * Returns allocated buffer; sets *out_nelem to element count.
 * Caller must free(). Returns NULL on error. */
static float *read_raw_float(const char *path, size_t *out_nelem)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    if (fsize <= 0 || fsize % sizeof(float) != 0) {
        fprintf(stderr, "Bad file size %ld for %s\n", fsize, path);
        fclose(f); return NULL;
    }

    size_t nelem = (size_t)fsize / sizeof(float);
    float *data = malloc(nelem * sizeof(float));
    if (!data) { fprintf(stderr, "OOM for %zu floats\n", nelem); fclose(f); return NULL; }

    size_t got = fread(data, sizeof(float), nelem, f);
    fclose(f);
    if (got != nelem) {
        fprintf(stderr, "Short read: got %zu / %zu\n", got, nelem);
        free(data); return NULL;
    }

    *out_nelem = nelem;
    return data;
}

/* ------------------------------------------------------------------ */
/*  Statistics helper                                                  */
/* ------------------------------------------------------------------ */

typedef struct { float min, max, mean; double rmse; } Stats;

static Stats compute_stats(const float *orig, const float *decomp, size_t n)
{
    Stats s = {orig[0], orig[0], 0.0f, 0.0};
    double sum = 0, sse = 0;
    for (size_t i = 0; i < n; i++) {
        if (orig[i] < s.min) s.min = orig[i];
        if (orig[i] > s.max) s.max = orig[i];
        sum += orig[i];
        double diff = (double)(orig[i] - decomp[i]);
        sse += diff * diff;
    }
    s.mean = (float)(sum / n);
    s.rmse = sqrt(sse / n);
    return s;
}

/* ------------------------------------------------------------------ */
/*  Per-field test                                                     */
/* ------------------------------------------------------------------ */

/*
 * Run one field through:
 *   1. noop  (reference, no compression)
 *   2. sz3   absolute error bound 1e-3
 *   3. zstd  lossless
 * Writes all three datasets to `file_id` with names like
 *   "<label>_noop", "<label>_sz3_1e3", "<label>_zstd"
 * and prints a quality/ratio summary.
 */
static int test_field(hid_t file_id,
                      const char *label,
                      const float *field,
                      size_t nelem,
                      int ndims,
                      const hsize_t *dims)
{
    printf("\n--- Field: %s  (%zu floats) ---\n", label, nelem);

    hid_t space_id = H5Screate_simple(ndims, dims, NULL);

    /* Helper: write one dataset, read back, report */
    struct {
        const char *suffix;
        const char *compressor;
        const char *json;
    } runs[] = {
        { "noop",     NULL,    NULL },
        { "sz3_1e3",  "sz3",
          "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}" },
        { "sz3_1e4",  "sz3",
          "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-4}" },
        { "zstd",     "zstd",
          "{\"zstd:clevel\":3}" },
    };
    int nruns = (int)(sizeof(runs) / sizeof(runs[0]));

    float *rbuf = malloc(nelem * sizeof(float));
    if (!rbuf) { fprintf(stderr, "OOM rbuf\n"); H5Sclose(space_id); return -1; }

    for (int r = 0; r < nruns; r++) {
        char dsname[256];
        snprintf(dsname, sizeof(dsname), "%s_%s", label, runs[r].suffix);

        hid_t dcpl = make_dcpl(runs[r].compressor, runs[r].json);
        hid_t dset = H5Dcreate2(file_id, dsname, H5T_NATIVE_FLOAT, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "  H5Dcreate2 failed for %s\n", dsname);
            H5Pclose(dcpl);
            continue;
        }

        herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                                H5S_ALL, H5S_ALL, H5P_DEFAULT, field);
        H5Dclose(dset);
        H5Pclose(dcpl);
        if (wret < 0) { fprintf(stderr, "  H5Dwrite failed for %s\n", dsname); continue; }

        /* Read back */
        memset(rbuf, 0, nelem * sizeof(float));
        dset = H5Dopen2(file_id, dsname, H5P_DEFAULT);
        herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                               H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
        H5Dclose(dset);

        if (rret < 0) { fprintf(stderr, "  H5Dread failed for %s\n", dsname); continue; }

        Stats st = compute_stats(field, rbuf, nelem);
        printf("  %-20s  range=[%.4g, %.4g]  mean=%.4g  RMSE=%.4e\n",
               dsname, st.min, st.max, st.mean, st.rmse);

        /* Spot-check first 5 values */
        printf("    orig:  ");
        for (int i = 0; i < 5; i++) printf("%.5e ", field[i]);
        printf("\n");
        printf("    decomp:");
        for (int i = 0; i < 5; i++) printf("%.5e ", rbuf[i]);
        printf("\n");
    }

    free(rbuf);
    H5Sclose(space_id);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Miranda test                                                       */
/* ------------------------------------------------------------------ */

static int run_miranda_test(hid_t file_id)
{
    /* SDRBench Miranda fields — all 256x384x384 float32 */
    const char *fields[] = {
        "velocityx.f32",
        "velocityy.f32",
        "velocityz.f32",
        "pressure.f32",
        "density.f32",
        NULL
    };

    hsize_t dims[3] = { MIR_NX, MIR_NY, MIR_NZ };

    int any_loaded = 0;
    for (int fi = 0; fields[fi]; fi++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", MIRANDA_PATH, fields[fi]);

        size_t nelem = 0;
        float *data = read_raw_float(path, &nelem);
        if (!data) {
            fprintf(stderr, "Skipping %s (not found or unreadable)\n", path);
            continue;
        }

        /* Verify expected element count */
        if (nelem != MIR_NELEM) {
            fprintf(stderr,
                    "WARNING: %s has %zu floats, expected %zu — "
                    "check Miranda dimensions\n",
                    fields[fi], nelem, MIR_NELEM);
            /* Use actual size as a 1-D dataset rather than skip */
            hsize_t dims1d[1] = { (hsize_t)nelem };

            /* Strip .f32 extension for the HDF5 dataset label */
            char label[128];
            strncpy(label, fields[fi], sizeof(label) - 1);
            char *dot = strrchr(label, '.');
            if (dot) *dot = '\0';

            test_field(file_id, label, data, nelem, 1, dims1d);
            free(data);
            continue;
        }

        any_loaded = 1;
        char label[128];
        strncpy(label, fields[fi], sizeof(label) - 1);
        char *dot = strrchr(label, '.');
        if (dot) *dot = '\0';

        test_field(file_id, label, data, nelem, 3, dims);
        free(data);
    }

    if (!any_loaded) {
        fprintf(stderr,
                "\nERROR: No Miranda fields loaded from %s\n"
                "Check that the path exists and files are named *.f32\n"
                "Run: ls %s\n",
                MIRANDA_PATH, MIRANDA_PATH);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  HACC test (optional)                                               */
/* ------------------------------------------------------------------ */

#ifdef USE_HACC
static int run_hacc_test(hid_t file_id)
{
    /* HACC files are raw float32 but the particle count varies by snapshot.
     * We read the whole file and treat it as a 1-D array.               */
    const char *hacc_fields[] = {
        "xx", "yy", "zz",   /* positions */
        "vx", "vy", "vz",   /* velocities */
        NULL
    };

    int any = 0;
    for (int fi = 0; hacc_fields[fi]; fi++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.f32", HACC_PATH, hacc_fields[fi]);

        size_t nelem = 0;
        float *data = read_raw_float(path, &nelem);
        if (!data) {
            fprintf(stderr, "Skipping HACC field %s\n", hacc_fields[fi]);
            continue;
        }
        any = 1;

        hsize_t dims1d[1] = { (hsize_t)nelem };
        char label[128];
        snprintf(label, sizeof(label), "hacc_%s", hacc_fields[fi]);
        test_field(file_id, label, data, nelem, 1, dims1d);
        free(data);
    }

    if (!any) {
        fprintf(stderr,
                "No HACC fields found under %s\n"
                "Run: ls %s/\n",
                HACC_PATH, HACC_PATH);
        return -1;
    }
    return 0;
}
#endif /* USE_HACC */

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("SDRBench VOL test starting\n"); fflush(stdout);
    register_vol_properties();

    hid_t file_id = H5Fcreate("sdrbench_test.h5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    int rc = 0;

    printf("\n=== Miranda (256x384x384 turbulence) ===\n");
    rc |= run_miranda_test(file_id);

#ifdef USE_HACC
    printf("\n=== HACC cosmological simulation ===\n");
    rc |= run_hacc_test(file_id);
#endif

    H5Fclose(file_id);

    if (rc == 0)
        printf("\nSDRBench VOL test PASSED.\n");
    else
        printf("\nSDRBench VOL test FINISHED WITH ERRORS (see above).\n");

    return rc;
}