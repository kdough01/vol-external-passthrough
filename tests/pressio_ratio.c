/* pressio_ratio.c -- ground truth for "what SHOULD this compress to?"
 *
 * Links against the same libpressio the VOL uses, so any disagreement is in
 * the VOL, not in the library.
 *
 * Build:
 *   LP=$HOME/libpressio_cuda/.spack-env/view
 *   gcc -O2 -std=c11 pressio_ratio.c -o pressio_ratio \
 *       -I$LP/include -L$LP/lib64 -L$LP/lib -llibpressio -lm \
 *       -Wl,-rpath,$LP/lib64 -Wl,-rpath,$LP/lib
 *
 * Run (dims in HDF5 C order, slowest first -- reversal is done internally):
 *   M=/lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384
 *   ./pressio_ratio $M/density.d64 256 384 384 sz3 1e-3
 *   ./pressio_ratio $M/density.d64 256 384 384 sz3 1e-6
 *
 * Sixth arg is the abs error bound; optional seventh is pressio:nthreads.
 * Compare nthreads=1 against nthreads=8: if the ratio drops, SZ3's OpenMP
 * path is your missing 2.5x and the VOL config is setting nthreads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libpressio/libpressio.h>

int
main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr,
            "usage: %s <file> <d0> <d1> <d2> <compressor> <abs_eb> [nthreads]\n"
            "       dims in HDF5 C order (slowest first)\n", argv[0]);
        return 2;
    }

    const char *path   = argv[1];
    size_t      h5d[3] = { strtoull(argv[2], NULL, 10),
                           strtoull(argv[3], NULL, 10),
                           strtoull(argv[4], NULL, 10) };
    const char *cid    = argv[5];
    double      eb     = strtod(argv[6], NULL);
    unsigned    nthr   = (argc > 7) ? (unsigned)strtoul(argv[7], NULL, 10) : 0;

    const size_t nelem  = h5d[0] * h5d[1] * h5d[2];
    const size_t nbytes = nelem * sizeof(double);

    /* libpressio wants fastest-varying FIRST */
    size_t pd[3] = { h5d[2], h5d[1], h5d[0] };

    double *raw = malloc(nbytes);
    if (!raw) { perror("malloc"); return 1; }

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    if (fread(raw, 1, nbytes, f) != nbytes) {
        fprintf(stderr, "short read: expected %zu bytes from %s\n", nbytes, path);
        return 1;
    }
    fclose(f);

    struct pressio *lib = pressio_instance();
    struct pressio_compressor *c = pressio_get_compressor(lib, cid);
    if (!c) {
        fprintf(stderr, "compressor '%s' unavailable. have: %s\n",
                cid, pressio_supported_compressors());
        return 1;
    }

    struct pressio_options *o = pressio_options_new();
    pressio_options_set_double(o, "pressio:abs", eb);
    if (nthr > 0)
        pressio_options_set_uinteger(o, "pressio:nthreads", nthr);
    if (pressio_compressor_set_options(c, o)) {
        fprintf(stderr, "set_options failed: %s\n", pressio_compressor_error_msg(c));
        return 1;
    }
    pressio_options_free(o);

    /* Print exactly what the codec is configured with, for diffing against
     * the VOL's dump. */
    {
        struct pressio_options *eff = pressio_compressor_get_options(c);
        char *s = pressio_options_to_string(eff);
        printf("=== %s  abs=%g  nthreads=%s\n", cid, eb, nthr ? argv[7] : "(default)");
        printf("dims: hdf5=[%zu,%zu,%zu] -> pressio=[%zu,%zu,%zu]\n",
               h5d[0], h5d[1], h5d[2], pd[0], pd[1], pd[2]);
        printf("--- effective options ---\n%s\n", s ? s : "(null)");
        free(s);
        pressio_options_free(eff);
    }

    struct pressio_data *in  = pressio_data_new_nonowning(pressio_double_dtype,
                                                          raw, 3, pd);
    struct pressio_data *cmp = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
    struct pressio_data *dec = pressio_data_new_owning(pressio_double_dtype, 3, pd);

    if (pressio_compressor_compress(c, in, cmp)) {
        fprintf(stderr, "compress failed: %s\n", pressio_compressor_error_msg(c));
        return 1;
    }

    size_t csize = 0;
    (void)pressio_data_ptr(cmp, &csize);

    if (pressio_compressor_decompress(c, cmp, dec)) {
        fprintf(stderr, "decompress failed: %s\n", pressio_compressor_error_msg(c));
        return 1;
    }

    /* Same metrics the filter harness prints, so the lines are comparable. */
    double *dp = (double *)pressio_data_ptr(dec, NULL);
    double sse = 0.0, maxae = 0.0;
    for (size_t i = 0; i < nelem; i++) {
        double d = raw[i] - dp[i];
        double a = fabs(d);
        sse += d * d;
        if (a > maxae) maxae = a;
    }

    printf("--- result ---\n");
    printf("orig=%zu B (%.2f MiB)  comp=%zu B (%.2f MiB)\n",
           nbytes, nbytes / 1048576.0, csize, csize / 1048576.0);
    printf("ratio=%.2fx  RMSE=%.3e  maxae=%.3e\n",
           (double)nbytes / (double)csize, sqrt(sse / (double)nelem), maxae);

    pressio_data_free(in);
    pressio_data_free(cmp);
    pressio_data_free(dec);
    pressio_compressor_release(c);
    pressio_release(lib);
    free(raw);
    return 0;
}
