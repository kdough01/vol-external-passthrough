/* ============================================================================
 * bench_filter_timing.c   --  APPROACH 3 of 3: HDF5 H5Z FILTER
 * ----------------------------------------------------------------------------
 * Runs a compressor through HDF5's built-in chunked filter pipeline. This is the
 * architectural comparator for your VOL: same file format, same HDF5 API, but
 * compression happens per-chunk inside H5Dwrite via an H5Z filter instead of in
 * a passthrough VOL.
 *
 * Two filter backends:
 *   --deflate         built-in gzip (H5Pset_deflate). Always available. A CPU
 *                     lossless comparator (compressor differs from sz3/cuszp, so
 *                     use it for the "CPU overhead comparable" claim, not for
 *                     ratio/fidelity).
 *   --libpressio      the libpressio H5Z filter, so the SAME codec (sz3/cuszp)
 *                     runs here as in the VOL. This is the apples-to-apples
 *                     architecture comparison. Requires the filter plugin on
 *                     HDF5_PLUGIN_PATH; set its id + config (see TODO).
 *
 * TIMING ASYMMETRY (important, and worth a sentence in the paper):
 *   In the filter path, compression and I/O are INTERLEAVED per chunk INSIDE
 *   H5Dwrite, driven by HDF5. You get TOTAL cleanly. You can only get COMPRESS
 *   by instrumenting INSIDE the filter callback (accumulate into the thread-local
 *   report, exactly like the VOL). You CANNOT cleanly separate IO, because HDF5
 *   owns the chunk writes. So report TOTAL and (if instrumented) COMPRESS, and
 *   treat TOTAL-COMPRESS as "HDF5 I/O + pipeline". Your VOL gives a cleaner
 *   3-way split than filters can -- that is itself a result.
 *
 * Build:
 *   h5cc -O2 -std=gnu11 bench_filter_timing.c -o bench_filter_timing
 * ==========================================================================*/
#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BENCH_DATASETS_ENABLE_HDF5
#include "bench_datasets.h"
#include "bench_compressors.h"
#include "bench_timing.h"

/* <<< CONFIRM: the libpressio H5Z filter id registered by robertu94's plugin.
 * There is no universal well-known number; set it to what H5Zget_filter_info /
 * the plugin reports on your build, or read it from an env var. */
#ifndef LIBPRESSIO_H5Z_FILTER_ID
#define LIBPRESSIO_H5Z_FILTER_ID ((H5Z_filter_t)32026)  /* placeholder */
#endif

typedef enum { FILTER_DEFLATE, FILTER_LIBPRESSIO } filter_backend_t;

/* Configure the dataset-creation plist: chunking (required for filters) + codec.
 * Default chunk = the whole field (one chunk). Experiment E4 sweeps this. */
static hid_t make_dcpl(const bench_dataset_t *d, filter_backend_t backend,
                       const bench_compressor_t *c) {
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, chunk);            /* single chunk = full dims */
    H5Pset_chunk(dcpl, d->rank, chunk);

    if (backend == FILTER_DEFLATE) {
        H5Pset_deflate(dcpl, 6);               /* gzip level 6 */
    } else {
        /* libpressio filter: pass the codec config through. The exact cd_values
         * / property mechanism is plugin-specific -- see the plugin's README.
         * Here we register the filter as mandatory with no cd_values and assume
         * the codec+options are supplied out-of-band (property or env). TODO. */
        (void)c;
        unsigned int cd_values[1] = {0};
        H5Pset_filter(dcpl, LIBPRESSIO_H5Z_FILTER_ID, H5Z_FLAG_MANDATORY,
                      0, cd_values);
    }
    return dcpl;
}

static size_t load_field(const bench_dataset_t *d, void **out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = malloc(nbytes);
    if (!buf) return 0;
    FILE *f = fopen(d->path, "rb");
    if (!f) { free(buf); return 0; }
    size_t got = fread(buf, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) { free(buf); return 0; }
    *out = buf;
    return nbytes;
}

static int run_one(const bench_dataset_t *d, filter_backend_t backend,
                   const bench_compressor_t *c, const char *h5path, FILE *csv) {
    void *hbuf = NULL;
    if (!load_field(d, &hbuf)) { fprintf(stderr, "skip %s\n", d->name); return -1; }

    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t htype = bench_dataset_h5type(d);
    const char *cname = (backend == FILTER_DEFLATE) ? "deflate" : c->name;

    /* ---- WRITE ---- */
    char wlabel[160];
    snprintf(wlabel, sizeof(wlabel), "%s/%s/filter/write", d->name, cname);
    bench_report_t wr; bench_report_reset(&wr, wlabel);

    hid_t dcpl  = make_dcpl(d, backend, c);
    hid_t file  = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);
    hid_t dset  = H5Dcreate2(file, d->name, htype, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    /* TOTAL only cleanly available here; a libpressio filter that accumulates
     * into bench_tls_get() would also fill COMPRESS (see timer_placement.md). */
    bench_tls_set(&wr);
    bench_scope_t sw = bench_scope_begin(&wr, BENCH_PHASE_TOTAL);
    H5Dwrite(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    bench_scope_end(&sw);
    bench_tls_set(NULL);

    H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl);

    hsize_t fsize = 0; H5Fget_filesize(file, &fsize);   /* for E4 file-size */
    H5Fclose(file);
    bench_report_derive_overhead(&wr);
    bench_report_csv_row(csv, &wr);
    fprintf(stderr, "%-30s file=%.1f MiB (raw %.1f MiB)\n", wlabel,
            fsize / (1024.0 * 1024.0), bench_num_bytes(d) / (1024.0 * 1024.0));

    /* ---- READ ---- */
    char rlabel[160];
    snprintf(rlabel, sizeof(rlabel), "%s/%s/filter/read", d->name, cname);
    bench_report_t rd; bench_report_reset(&rd, rlabel);

    void *rbuf = malloc(bench_num_bytes(d));
    file = H5Fopen(h5path, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, d->name, H5P_DEFAULT);

    bench_tls_set(&rd);
    bench_scope_t sr = bench_scope_begin(&rd, BENCH_PHASE_TOTAL);
    H5Dread(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    bench_scope_end(&sr);
    bench_tls_set(NULL);

    H5Dclose(dset); H5Fclose(file);
    bench_report_derive_overhead(&rd);
    bench_report_csv_row(csv, &rd);

    free(rbuf); free(hbuf);
    return 0;
}

int main(int argc, char **argv) {
    const char *h5path  = "bench_filter.h5";
    const char *csvpath = "results_filter.csv";
    filter_backend_t backend = FILTER_DEFLATE;
    const char *only = NULL;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--deflate"))    backend = FILTER_DEFLATE;
        else if (!strcmp(argv[i], "--libpressio")) backend = FILTER_LIBPRESSIO;
        else if (!strcmp(argv[i], "--comp") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--out")  && i + 1 < argc) csvpath = argv[++i];
        else if (!strcmp(argv[i], "--h5")   && i + 1 < argc) h5path = argv[++i];
    }

    /* Verify the chosen filter is actually available before running. */
    if (backend == FILTER_DEFLATE && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0)
        fprintf(stderr, "WARNING: deflate filter not available in this HDF5\n");
    if (backend == FILTER_LIBPRESSIO && H5Zfilter_avail(LIBPRESSIO_H5Z_FILTER_ID) <= 0)
        fprintf(stderr, "WARNING: libpressio filter id %d not available "
                        "(set HDF5_PLUGIN_PATH / LIBPRESSIO_H5Z_FILTER_ID)\n",
                        (int)LIBPRESSIO_H5Z_FILTER_ID);

    bench_datasets_validate();
    FILE *csv = fopen(csvpath, "w");
    if (!csv) { perror("csv"); return 1; }
    bench_report_csv_header(csv);

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (access(d->path, R_OK) != 0) continue;
        if (backend == FILTER_DEFLATE) {
            run_one(d, backend, NULL, h5path, csv);
        } else {
            for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
                const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
                if (only && strcmp(only, c->name)) continue;
                run_one(d, backend, c, h5path, csv);
            }
        }
    }
    fclose(csv);
    fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}