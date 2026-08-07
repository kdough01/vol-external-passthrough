/* tests/probe_sperr_prog.cc
 *
 * libpressio compress -> sperr_trunc_3d -> libpressio decompress
 *
 *   usage: ./probe_sperr_prog [pct] [abs_tolerance]
 *          omit pct to sweep 100/50/25/12/6
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <libpressio/libpressio.h>
#include <SPERR_C_API.h>

static double
rmse_of(const float *a, const float *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return sqrt(s / (double)n);
}

int
main(int argc, char **argv)
{
    const double abs_tol = (argc > 2) ? atof(argv[2]) : 1e-3;

    const size_t nx = 128, ny = 128, nz = 128;
    const size_t n  = nx * ny * nz;
    size_t dims[3]  = { nx, ny, nz };        /* libpressio order, fastest first */

    printf("supported: %s\n\n", pressio_supported_compressors());

    std::vector<float> orig(n), recon(n);
    for (size_t k = 0; k < nz; k++)
        for (size_t j = 0; j < ny; j++)
            for (size_t i = 0; i < nx; i++)
                orig[(k * ny + j) * nx + i] =
                    (float)(sin(i * 0.05) * cos(j * 0.03) + 0.3 * sin(k * 0.07));

    struct pressio *lib = pressio_instance();
    struct pressio_compressor *c = pressio_get_compressor(lib, "sperr");
    if (!c) { fprintf(stderr, "sperr not available\n"); return 1; }

    /* One chunk covering the whole volume: a truncated stream is then a plain
     * prefix. Multi-chunk streams interleave, so the bytes needed would be
     * strided instead. */
    {
        struct pressio_options *o = pressio_options_new();
        size_t cd[1] = { 3 };
        uint64_t chunk[3] = { nx, ny, nz };
        struct pressio_data *cdata =
            pressio_data_new_nonowning(pressio_uint64_dtype, chunk, 1, cd);

        pressio_options_set_double(o, "pressio:abs", abs_tol);
        pressio_options_set_data(o, "sperr:chunks", cdata);
        if (pressio_compressor_set_options(c, o))
            fprintf(stderr, "set_options: %s\n", pressio_compressor_error_msg(c));
        pressio_data_free(cdata);
        pressio_options_free(o);
    }

    {
        struct pressio_options *o = pressio_compressor_get_options(c);
        char *s = pressio_options_to_string(o);
        printf("=== sperr options ===\n%s\n\n", s ? s : "(null)");
        free(s);
        pressio_options_free(o);
    }

    struct pressio_data *input =
        pressio_data_new_nonowning(pressio_float_dtype, orig.data(), 3, dims);
    struct pressio_data *comp =
        pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    if (pressio_compressor_compress(c, input, comp)) {
        fprintf(stderr, "compress failed: %s\n", pressio_compressor_error_msg(c));
        return 1;
    }

    size_t csize = 0;
    void  *cbuf  = pressio_data_ptr(comp, &csize);
    printf("compressed %zu B -> %zu B (%.2fx), abs=%g\n",
           n * sizeof(float), csize,
           (double)(n * sizeof(float)) / (double)csize, abs_tol);

    {
        std::vector<float> z(n, 0.0f);
        printf("reference rmse(zeros vs orig) = %.6e\n\n",
               rmse_of(orig.data(), z.data(), n));
    }

    std::vector<unsigned> pcts;
    if (argc > 1) pcts.push_back((unsigned)atoi(argv[1]));
    else { pcts.push_back(100); pcts.push_back(50); pcts.push_back(25);
           pcts.push_back(12);  pcts.push_back(6); }

    printf("%-6s %-12s %-12s %-5s %-5s %-14s %-10s %s\n",
           "pct", "fed_bytes", "trunc_bytes", "trc", "rc", "rmse", "wrote", "err");

    for (size_t idx = 0; idx < pcts.size(); idx++) {
        unsigned pct = pcts[idx];

        /* Simulate the VOL read: only a prefix of the stream is available,
         * in its own exactly-sized allocation. Over-read slightly, the way
         * vol_container_plan will, since sperr_trunc_3d tolerates a stream
         * longer than it needs but not shorter. */
        size_t fed = (csize * pct) / 100 + 4096;
        if (fed > csize) fed = csize;

        void *prefix = malloc(fed);
        if (!prefix) { fprintf(stderr, "oom\n"); return 1; }
        memcpy(prefix, cbuf, fed);

        void  *ts   = NULL;
        size_t tlen = 0;
        int trc = C_API::sperr_trunc_3d(prefix, fed, pct, &ts, &tlen);

        int    rc = -1;
        double r  = -2.0;
        const float *rp = NULL;

        if (trc == 0 && ts && tlen) {
            size_t cd[1] = { tlen };
            struct pressio_data *in =
                pressio_data_new_nonowning(pressio_byte_dtype, ts, 1, cd);
            struct pressio_data *out =
                pressio_data_new_nonowning(pressio_float_dtype, recon.data(), 3, dims);

            memset(recon.data(), 0, n * sizeof(float));
            rc = pressio_compressor_decompress(c, in, out);

            size_t osz = 0;
            rp = (const float *)pressio_data_ptr(out, &osz);
            if (!rc && rp && osz == n * sizeof(float))
                r = rmse_of(orig.data(), rp, n);

            printf("%-6u %-12zu %-12zu %-5d %-5d %-14.6e %-10s %s\n",
                   pct, fed, tlen, trc, rc, r,
                   (rp == recon.data()) ? "in-place" : "REALLOC",
                   rc ? pressio_compressor_error_msg(c) : "");

            pressio_data_free(in);
            pressio_data_free(out);
        } else {
            printf("%-6u %-12zu %-12zu %-5d %-5s %-14s %-10s %s\n",
                   pct, fed, tlen, trc, "-", "-", "-", "sperr_trunc_3d failed");
        }

        fflush(stdout);
        free(ts);
        free(prefix);
    }

    pressio_data_free(input);
    pressio_data_free(comp);
    pressio_compressor_release(c);
    pressio_release(lib);
    return 0;
}