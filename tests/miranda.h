#ifndef TEST_MIRANDA_COMMON_H
#define TEST_MIRANDA_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "hdf5.h"

#define MIRANDA_PATH \
    "/lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384"

#define MIR_NX    256
#define MIR_NY    384
#define MIR_NZ    384
#define MIR_NELEM ((size_t)MIR_NX * MIR_NY * MIR_NZ)

#define DEFAULT_COMPRESSOR "noop"
#define DEFAULT_LEVEL 1

static double elapsed_ms(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec  - t0.tv_sec)  * 1000.0 +
           (t1.tv_nsec - t0.tv_nsec) / 1e6;
}

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
    char buf[64];
    strncpy(buf, compressor ? compressor : DEFAULT_COMPRESSOR, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    H5Pset(dcpl, "pressio:compressor", buf);
    if (json_opts) {
        char jbuf[4096] = "";
        strncpy(jbuf, json_opts, sizeof(jbuf) - 1);
        H5Pset(dcpl, "vol:options_json", jbuf);
    }
    return dcpl;
}

static double *read_raw_double(const char *path, size_t *out_nelem)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    if (fsize <= 0 || fsize % (long)sizeof(double) != 0) {
        fprintf(stderr, "Bad file size %ld for %s\n", fsize, path);
        fclose(f); return NULL;
    }

    size_t nelem = (size_t)fsize / sizeof(double);
    double *data = malloc(nelem * sizeof(double));
    if (!data) { fprintf(stderr, "OOM for %zu doubles\n", nelem); fclose(f); return NULL; }

    size_t got = fread(data, sizeof(double), nelem, f);
    fclose(f);
    if (got != nelem) {
        fprintf(stderr, "Short read: got %zu / %zu\n", got, nelem);
        free(data); return NULL;
    }

    *out_nelem = nelem;
    return data;
}

typedef struct { double min, max, mean, rmse; } Stats;

static Stats compute_stats(const double *orig, const double *decomp, size_t n)
{
    Stats s = { orig[0], orig[0], 0.0, 0.0 };
    double sum = 0, sse = 0;
    for (size_t i = 0; i < n; i++) {
        if (orig[i] < s.min) s.min = orig[i];
        if (orig[i] > s.max) s.max = orig[i];
        sum += orig[i];
        double diff = orig[i] - decomp[i];
        sse += diff * diff;
    }
    s.mean = sum / (double)n;
    s.rmse = sqrt(sse / (double)n);
    return s;
}

#endif