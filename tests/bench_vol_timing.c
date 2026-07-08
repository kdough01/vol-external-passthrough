/* ============================================================================
 * bench_vol_timing_example.c
 * ----------------------------------------------------------------------------
 * Reference harness showing how bench_datasets.h + bench_timing.h wire together
 * for the "total write / total read time" experiments across every dataset.
 *
 * The VOL adaptor is selected transparently via environment, so this harness is
 * PLAIN HDF5 -- the same code benchmarks the native path, the H5Z-filter path,
 * and your passthrough VOL. That transparency is itself one of the paper's
 * claims (see experiment_design.md).
 *
 *   # your VOL adaptor
 *   HDF5_PLUGIN_PATH=... \
 *   HDF5_VOL_CONNECTOR="pass_through_ext under_vol=0;under_info={}" \
 *   ./bench_vol_timing_example out.h5 results_vol.csv
 *
 *   # native (no connector) -> baseline
 *   ./bench_vol_timing_example out.h5 results_native.csv
 *
 *   # H5Z filter path -> set the filter in code / via a plist variant
 *
 * Build:
 *   h5cc -O2 -std=c11 bench_vol_timing_example.c -o bench_vol_timing_example
 *   (or: cc -O2 -std=c11 bench_vol_timing_example.c -lhdf5 -o ...)
 *
 * COMPRESS and IO are measured INSIDE the connector; this harness measures
 * TOTAL and reads back the connector's per-op breakdown via the thread-local
 * report (Note 3, option b). Against native/filters the connector breakdown is
 * simply absent and overhead is derived as TOTAL - COMPRESS - IO = TOTAL.
 * ==========================================================================*/
#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BENCH_DATASETS_ENABLE_HDF5   /* enable the H5T_* / hsize_t helpers */
#include "bench_datasets.h"
#include "bench_timing.h"

/* Load one raw binary field into a malloc'd buffer. Returns bytes read. */
static size_t load_field(const bench_dataset_t *d, void **out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = malloc(nbytes);
    if (!buf) { fprintf(stderr, "OOM for %s\n", d->name); return 0; }
    FILE *f = fopen(d->path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", d->path); free(buf); return 0; }
    size_t got = fread(buf, 1, nbytes, f);
    fclose(f);
    if (got != nbytes) {
        fprintf(stderr, "%s: read %zu of %zu bytes\n", d->name, got, nbytes);
        free(buf); return 0;
    }
    *out = buf;
    return nbytes;
}

/* One dataset: write then read, timing TOTAL around the H5D calls. */
static int run_one(const bench_dataset_t *d, const char *h5path, FILE *csv) {
    void *hbuf = NULL;
    if (!load_field(d, &hbuf)) return -1;

    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t htype = bench_dataset_h5type(d);

    /* ---- WRITE ---------------------------------------------------------- */
    bench_report_t wr; bench_report_reset(&wr, NULL);
    char wlabel[128];
    snprintf(wlabel, sizeof(wlabel), "%s/write", d->name);
    wr.label = wlabel;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);   /* VOL comes from env */
    hid_t file = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);
    hid_t dset  = H5Dcreate2(file, d->name, htype, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Note 3(b): expose this op's report to the connector's callbacks. */
    bench_tls_set(&wr);
    bench_scope_t sw = bench_scope_begin(&wr, BENCH_PHASE_TOTAL);
    herr_t status = H5Dwrite(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    bench_scope_end(&sw);
    bench_tls_set(NULL);
    if (status < 0) fprintf(stderr, "%s: H5Dwrite failed\n", d->name);

    bench_report_derive_overhead(&wr);
    bench_report_csv_row(csv, &wr);

    H5Dclose(dset); H5Sclose(space); H5Fclose(file);

    /* ---- READ ----------------------------------------------------------- */
    bench_report_t rd; bench_report_reset(&rd, NULL);
    char rlabel[128];
    snprintf(rlabel, sizeof(rlabel), "%s/read", d->name);
    rd.label = rlabel;

    void *rbuf = malloc(bench_num_bytes(d));
    file = H5Fopen(h5path, H5F_ACC_RDONLY, fapl);
    dset = H5Dopen2(file, d->name, H5P_DEFAULT);

    bench_tls_set(&rd);
    bench_scope_t sr = bench_scope_begin(&rd, BENCH_PHASE_TOTAL);
    status = H5Dread(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    bench_scope_end(&sr);
    bench_tls_set(NULL);
    if (status < 0) fprintf(stderr, "%s: H5Dread failed\n", d->name);

    bench_report_derive_overhead(&rd);
    bench_report_csv_row(csv, &rd);

    H5Dclose(dset); H5Fclose(file); H5Pclose(fapl);
    free(rbuf); free(hbuf);
    return 0;
}

int main(int argc, char **argv) {
    const char *h5path = (argc > 1) ? argv[1] : "bench_out.h5";
    const char *csvpath = (argc > 2) ? argv[2] : "results.csv";

    /* Fail loudly on any misnamed cluster path before doing real work. */
    bench_datasets_validate();

    FILE *csv = fopen(csvpath, "w");
    if (!csv) { perror("csv"); return 1; }
    bench_report_csv_header(csv);

    for (int i = 0; i < BENCH_NUM_DATASETS; ++i) {
        const bench_dataset_t *d = &BENCH_DATASETS[i];
        if (access(d->path, R_OK) != 0) {
            fprintf(stderr, "skip %s (path not readable)\n", d->name);
            continue;
        }
        run_one(d, h5path, csv);
    }
    fclose(csv);
    fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}

/* ----------------------------------------------------------------------------
 * SKETCH: what the connector does INSIDE its dataset_write callback, so that
 * COMPRESS (GPU-correct) and IO fold into the same report this harness set via
 * bench_tls_set(). Lives in compress.cc, compiled with nvcc + BENCH_TIMING_WITH_CUDA.
 *
 *   bench_report_t *rep = bench_tls_get();   // may be NULL under native/filter
 *
 *   // --- compressor (GPU): events on the SAME stream cuszp uses ----------
 *   #ifdef BENCH_TIMING_WITH_CUDA
 *     bench_gpu_timer_t g = bench_gpu_timer_create(stream);  // your userptr stream
 *     bench_gpu_timer_start(&g);
 *     pressio_compress(compressor, in_data, &out_data);      // enqueues on stream
 *     bench_gpu_timer_stop(&g);
 *     bench_report_add_gpu(rep, BENCH_PHASE_COMPRESS, bench_gpu_timer_elapsed_ms(&g));
 *     bench_gpu_timer_destroy(&g);
 *   #else
 *     bench_scope_t sc = bench_scope_begin(rep, BENCH_PHASE_COMPRESS);
 *     pressio_compress(compressor, in_data, &out_data);      // CPU codec
 *     bench_scope_end(&sc);
 *   #endif
 *
 *   // --- IO: the underlying H5VLdataset_write of the compressed bytes -----
 *   bench_scope_t si = bench_scope_begin(rep, BENCH_PHASE_IO);
 *   H5VLdataset_write(under, ...);
 *   bench_scope_end(&si);
 *
 *   // TOTAL is measured by the harness; OVERHEAD = TOTAL - COMPRESS - IO.
 * ------------------------------------------------------------------------- */