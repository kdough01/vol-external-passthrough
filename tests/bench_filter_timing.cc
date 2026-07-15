/* ============================================================================
 * bench_filter_timing.cc  --  APPROACH 3 of 3: HDF5 H5Z FILTER
 * ----------------------------------------------------------------------------
 * Runs a compressor through HDF5's chunked filter pipeline: same file format and
 * HDF5 API as the VOL, but compression happens per-chunk inside H5Dwrite via an
 * H5Z filter. Architectural comparator for the VOL.
 *
 * Backends:
 *   --deflate      built-in gzip (H5Pset_deflate). Always available. CPU
 *                  lossless comparator (different codec from sz3/cuszp, so use
 *                  for the "CPU overhead comparable" claim, not ratio/fidelity).
 *   --libpressio   the libpressio H5Z filter, so the SAME codec runs here as in
 *                  the VOL. Apples-to-apples. Needs the filter plugin on
 *                  HDF5_PLUGIN_PATH; set its id + config (see TODO).
 *
 * TIMING ASYMMETRY (worth a sentence in the paper): in the filter path compress
 * and I/O are INTERLEAVED per chunk inside H5Dwrite, driven by HDF5. This
 * harness measures TOTAL cleanly (steady_clock around H5Dwrite/H5Dread). COMPRESS
 * is only obtainable by instrumenting INSIDE the filter callback; I/O cannot be
 * cleanly separated because HDF5 owns the chunk writes. So this emits TOTAL only.
 * The VOL's clean 3-way split is itself a result.
 *
 * Build:
 *   h5c++ -O2 -std=c++17 bench_filter_timing.cc -o bench_filter_timing
 * ==========================================================================*/
#include <hdf5.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#define BENCH_CONFIG_ENABLE_HDF5
#include "bench_config.h"
#include "bench_timing.h"

/* <<< CONFIRM: the libpressio H5Z filter id registered by robertu94's plugin.
 * No universal well-known number; set it to what the plugin reports on your
 * build, or read it from an env var. */
#ifndef LIBPRESSIO_H5Z_FILTER_ID
#define LIBPRESSIO_H5Z_FILTER_ID ((H5Z_filter_t)32026)  /* placeholder */
#endif

typedef enum { FILTER_DEFLATE, FILTER_LIBPRESSIO } filter_backend_t;

/* dataset-creation plist: chunking (required for filters) + codec.
 * Default chunk = whole field (one chunk). Experiment E4 sweeps this. */
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
         * Registered mandatory with no cd_values here; codec+options assumed
         * supplied out-of-band (property or env). TODO. */
        (void)c;
        unsigned int cd_values[1] = {0};
        H5Pset_filter(dcpl, LIBPRESSIO_H5Z_FILTER_ID, H5Z_FLAG_MANDATORY,
                      0, cd_values);
    }
    return dcpl;
}

static size_t load_field(const bench_dataset_t *d, void **out) {
    size_t nbytes = bench_num_bytes(d);
    void *buf = std::malloc(nbytes);
    if (!buf) return 0;
    FILE *f = std::fopen(d->path, "rb");
    if (!f) { std::free(buf); return 0; }
    size_t got = std::fread(buf, 1, nbytes, f);
    std::fclose(f);
    if (got != nbytes) { std::free(buf); return 0; }
    *out = buf;
    return nbytes;
}

static int run_one(const bench_dataset_t *d, filter_backend_t backend,
                   const bench_compressor_t *c, const char *h5path, FILE *csv) {
    void *hbuf = NULL;
    if (!load_field(d, &hbuf)) { std::fprintf(stderr, "skip %s\n", d->name); return -1; }

    hsize_t dims[BENCH_MAX_RANK];
    bench_dataset_h5dims(d, dims);
    hid_t htype = bench_dataset_h5type(d);
    const char *cname = (backend == FILTER_DEFLATE) ? "deflate" : c->name;

    /* ---- WRITE ---- */
    hid_t dcpl  = make_dcpl(d, backend, c);
    hid_t file  = H5Fcreate(h5path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space = H5Screate_simple(d->rank, dims, NULL);
    hid_t dset  = H5Dcreate2(file, d->name, htype, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    BenchCpuTimer wt; wt.start();
    H5Dwrite(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, hbuf);
    double wms = wt.stop_ms();

    H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl);

    hsize_t fsize = 0; H5Fget_filesize(file, &fsize);     /* for E4 file-size */
    H5Fclose(file);

    double wratio = fsize ? (double)bench_num_bytes(d) / (double)fsize : 0.0;
    bench_csv_row(csv, d->name, cname, "filter", "write", "total",
                  wms, wratio, -1.0);
    std::fprintf(stderr, "%-30s file=%.1f MiB (raw %.1f MiB)\n",
                 (std::string(d->name) + "/" + cname).c_str(),
                 fsize / (1024.0 * 1024.0),
                 bench_num_bytes(d) / (1024.0 * 1024.0));

    /* ---- READ ---- */
    void *rbuf = std::malloc(bench_num_bytes(d));
    file = H5Fopen(h5path, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file, d->name, H5P_DEFAULT);

    BenchCpuTimer rt; rt.start();
    H5Dread(dset, htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    double rms = rt.stop_ms();

    H5Dclose(dset); H5Fclose(file);
    bench_csv_row(csv, d->name, cname, "filter", "read", "total",
                  rms, -1.0, -1.0);

    std::free(rbuf); std::free(hbuf);
    return 0;
}

int main(int argc, char **argv) {
    const char *h5path  = "bench_filter.h5";
    const char *csvpath = "results_filter.csv";
    filter_backend_t backend = FILTER_DEFLATE;
    const char *only = NULL;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--deflate"))    backend = FILTER_DEFLATE;
        else if (!std::strcmp(argv[i], "--libpressio")) backend = FILTER_LIBPRESSIO;
        else if (!std::strcmp(argv[i], "--comp") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--out")  && i + 1 < argc) csvpath = argv[++i];
        else if (!std::strcmp(argv[i], "--h5")   && i + 1 < argc) h5path = argv[++i];
    }

    if (backend == FILTER_DEFLATE && H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0)
        std::fprintf(stderr, "WARNING: deflate filter not available in this HDF5\n");
    if (backend == FILTER_LIBPRESSIO && H5Zfilter_avail(LIBPRESSIO_H5Z_FILTER_ID) <= 0)
        std::fprintf(stderr, "WARNING: libpressio filter id %d not available "
                             "(set HDF5_PLUGIN_PATH / LIBPRESSIO_H5Z_FILTER_ID)\n",
                             (int)LIBPRESSIO_H5Z_FILTER_ID);

    bench_datasets_validate();
    FILE *csv = std::fopen(csvpath, "w");
    if (!csv) { std::perror("csv"); return 1; }
    bench_csv_header(csv);

    for (int di = 0; di < BENCH_NUM_DATASETS; ++di) {
        const bench_dataset_t *d = &BENCH_DATASETS[di];
        if (access(d->path, R_OK) != 0) continue;
        if (backend == FILTER_DEFLATE) {
            run_one(d, backend, NULL, h5path, csv);
        } else {
            for (int ci = 0; ci < BENCH_NUM_COMPRESSORS; ++ci) {
                const bench_compressor_t *c = &BENCH_COMPRESSORS[ci];
                if (only && std::strcmp(only, c->name)) continue;
                run_one(d, backend, c, h5path, csv);
            }
        }
    }
    std::fclose(csv);
    std::fprintf(stderr, "wrote %s\n", csvpath);
    return 0;
}