/* tests/probe_sperr_trunc.cc
 *
 * Does a libpressio codec survive decompression from a truncated bitstream?
 *
 *   usage: ./probe_trunc [compressor] [fraction] [abs_tolerance]
 *
 *     compressor     default "mgard"   (also: sperr, sz3, zfp, ...)
 *     fraction       run a single fraction in this process; omit to run the
 *                    full descending sweep 1.0 -> 0.0625
 *     abs_tolerance  default 1e-3
 *
 * Each truncated prefix is copied into its own exactly-sized allocation, so a
 * codec that reads its length from its own header will fault or error rather
 * than silently reading past the end into still-valid heap. Run one fraction
 * per process under ASAN to rule that out for good:
 *
 *   for f in 1.0 0.5 0.25 0.125 0.0625; do ./probe_trunc mgard $f; done
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <libpressio/libpressio.h>

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
    const char  *cname = (argc > 1) ? argv[1] : "mgard";
    const double abs_tol = (argc > 3) ? atof(argv[3]) : 1e-3;

    /* 3D field; SPERR and MGARD both want 2D or 3D. */
    const size_t nx = 128, ny = 128, nz = 128;
    const size_t n  = nx * ny * nz;
    size_t dims[3]  = { nx, ny, nz };      /* libpressio order: fastest first */

    printf("supported: %s\n\n", pressio_supported_compressors());

    std::vector<float> orig(n), recon(n);
    for (size_t k = 0; k < nz; k++)
        for (size_t j = 0; j < ny; j++)
            for (size_t i = 0; i < nx; i++)
                orig[(k * ny + j) * nx + i] =
                    (float)(sin(i * 0.05) * cos(j * 0.03) + 0.3 * sin(k * 0.07));

    struct pressio *lib = pressio_instance();
    struct pressio_compressor *c = pressio_get_compressor(lib, cname);
    if (!c) {
        fprintf(stderr, "'%s' not available in this build\n", cname);
        return 1;
    }

    /* Dump the real option keys -- also the answer to "what are this codec's
     * knobs", which the docs do not cover for sperr or mgard. */
    {
        struct pressio_options *o = pressio_compressor_get_options(c);
        char *s = pressio_options_to_string(o);
        printf("=== %s options ===\n%s\n\n", cname, s ? s : "(null)");
        free(s);
        pressio_options_free(o);
    }

    /* Loose enough that the codec actually compresses. An abs bound below
     * float32's own precision (e.g. 1e-6 on O(1) data) makes codecs store
     * essentially raw, leaving nothing meaningful to truncate. */
    {
        struct pressio_options *o = pressio_options_new();
        pressio_options_set_double(o, "pressio:abs", abs_tol);
        if (pressio_compressor_set_options(c, o))
            fprintf(stderr, "set_options: %s\n", pressio_compressor_error_msg(c));
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
    printf("compressor=%s abs=%g\n", cname, abs_tol);
    printf("compressed %zu B -> %zu B (%.2fx)\n",
           n * sizeof(float), csize,
           (double)(n * sizeof(float)) / (double)csize);

    /* Reference: RMSE of an all-zero field. Any result equal to this means the
     * codec wrote nothing, not that it reconstructed badly. */
    {
        std::vector<float> z(n, 0.0f);
        printf("reference rmse(zeros vs orig) = %.6e\n\n",
               rmse_of(orig.data(), z.data(), n));
    }

    std::vector<double> fracs;
    if (argc > 2) fracs.push_back(atof(argv[2]));
    else for (double f = 1.0; f >= 0.0625; f /= 2) fracs.push_back(f);

    printf("%-8s %-12s %-4s %-14s %-10s %s\n",
           "frac", "bytes", "rc", "rmse", "wrote", "err");

    for (size_t idx = 0; idx < fracs.size(); idx++) {
        double f = fracs[idx];
        size_t trunc = (size_t)((double)csize * f);
        if (trunc == 0) continue;

        /* Exact-size copy: nothing valid lives past the end. */
        void *chunk = malloc(trunc);
        if (!chunk) { fprintf(stderr, "oom\n"); return 1; }
        memcpy(chunk, cbuf, trunc);

        size_t cd[1] = { trunc };
        struct pressio_data *in =
            pressio_data_new_nonowning(pressio_byte_dtype, chunk, 1, cd);
        struct pressio_data *out =
            pressio_data_new_nonowning(pressio_float_dtype, recon.data(), 3, dims);

        memset(recon.data(), 0, n * sizeof(float));
        int rc = pressio_compressor_decompress(c, in, out);

        /* Read from wherever the codec actually wrote -- several plugins
         * replace the output buffer instead of filling the one they are given,
         * in which case `recon` is untouched and measuring it is meaningless. */
        size_t osz = 0;
        const float *rp = (const float *)pressio_data_ptr(out, &osz);
        double r = -2.0;
        if (!rc && rp && osz == n * sizeof(float))
            r = rmse_of(orig.data(), rp, n);

        printf("%-8.4f %-12zu %-4d %-14.6e %-10s %s\n",
               f, trunc, rc, r,
               (rp == recon.data()) ? "in-place" : "REALLOC",
               rc ? pressio_compressor_error_msg(c) : "");
        fflush(stdout);

        pressio_data_free(in);
        pressio_data_free(out);
        free(chunk);
    }

    pressio_data_free(input);
    pressio_data_free(comp);
    pressio_compressor_release(c);
    pressio_release(lib);
    return 0;
}