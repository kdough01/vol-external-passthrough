/* tests/probe_sperr_trunc.cc  --  does libpressio's sperr survive truncation? */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <libpressio/libpressio.h>

static double rmse_of(const float *a, const float *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; s += d * d; }
    return sqrt(s / (double)n);
}

int main(int argc, char **argv)
{
    /* 3D field; SPERR wants 2D or 3D. Override with argv or load a real one. */
    const size_t nx = 128, ny = 128, nz = 128;
    const size_t n  = nx * ny * nz;
    size_t dims[3]  = { nx, ny, nz };   /* libpressio order: fastest first */

    printf("supported: %s\n\n", pressio_supported_compressors());

    std::vector<float> orig(n), recon(n);
    for (size_t k = 0; k < nz; k++)
      for (size_t j = 0; j < ny; j++)
        for (size_t i = 0; i < nx; i++)
          orig[(k*ny + j)*nx + i] =
              (float)(sin(i * 0.05) * cos(j * 0.03) + 0.3 * sin(k * 0.07));

    struct pressio *lib = pressio_instance();
    struct pressio_compressor *c = pressio_get_compressor(lib, "sperr");
    if (!c) { fprintf(stderr, "sperr not available in this build\n"); return 1; }

    /* Dump the real option keys -- this is also the answer to "what are
     * sperr's knobs", which the docs don't cover. */
    {
        struct pressio_options *o = pressio_compressor_get_options(c);
        char *s = pressio_options_to_string(o);
        printf("=== sperr options ===\n%s\n\n", s);
        free(s);
        pressio_options_free(o);
    }

    /* High quality so there is a long stream to truncate. If pressio:abs is
     * not honoured, switch to sperr's own rate/quality key from the dump. */
    /* Loose enough that MGARD actually compresses. 1e-6 on O(1) float32
     * data is below the type's own precision -- it stores raw. */
    {
        struct pressio_options *o = pressio_options_new();
        pressio_options_set_double(o, "pressio:abs", 1e-3);
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
    printf("compressed %zu B -> %zu B (%.2fx)\n\n",
           n * sizeof(float), csize, (double)(n * sizeof(float)) / (double)csize);

    printf("%-8s %-12s %-4s %-14s %s\n", "frac", "bytes", "rc", "rmse", "note");
    /* Reference: RMSE of an all-zero field, so "wrote nothing" is
     * distinguishable from "wrote something wrong". */
    {
        std::vector<float> z(n, 0.0f);
        printf("reference rmse(zeros vs orig) = %.6e\n\n",
               rmse_of(orig.data(), z.data(), n));
    }

    printf("%-8s %-12s %-4s %-14s %-12s %s\n",
           "frac", "bytes", "rc", "rmse", "wrote", "err");
    for (double f = 1.0; f >= 0.0625; f /= 2) {
        size_t trunc = (size_t)((double)csize * f);
        if (trunc == 0) continue;

        /* Exact-size copy. Nothing valid lives past the end, so a codec that
         * reads its length from its own header will fault or garbage rather
         * than silently succeeding. */
        void *chunk = malloc(trunc);
        memcpy(chunk, cbuf, trunc);

        size_t cd[1] = { trunc };
        struct pressio_data *in =
            pressio_data_new_nonowning(pressio_byte_dtype, chunk, 1, cd);
        struct pressio_data *out =
            pressio_data_new_nonowning(pressio_float_dtype, recon.data(), 3, dims);

        memset(recon.data(), 0, n * sizeof(float));
        int rc = pressio_compressor_decompress(c, in, out);

        size_t osz = 0;
        const float *rp = (const float *)pressio_data_ptr(out, &osz);
        double r = -2.0;
        if (!rc && rp && osz == n * sizeof(float))
            r = rmse_of(orig.data(), rp, n);

        printf("%-8.4f %-12zu %-4d %-14.6e %-12s %s\n",
               f, trunc, rc, r,
               (rp == recon.data()) ? "in-place" : "REALLOC",
               rc ? pressio_compressor_error_msg(c) : "");

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