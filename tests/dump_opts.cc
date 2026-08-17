/* dump_opts.cpp -- print every option a libpressio compressor exposes.
 *
 * Generic version of dump_cuszp_opts. One decisive question here: does the
 * sperr plugin let us set SPERR's internal chunk dimensions? If it does,
 * progressive truncation is safe on full-resolution data. If it does not,
 * SPERR's default 256^3 blocking splits every dataset in bench_config.h into
 * multiple internal chunks and a byte prefix of the bitstream is not decodable.
 *
 * Build:
 *   bash build_dump_opts.sh
 * or by hand:
 *   export LP_VIEW=/home/kdougherty/libpressio_cuda/.spack-env/view
 *   g++ -O2 -std=c++17 dump_opts.cpp -o dump_opts \
 *       -I$LP_VIEW/include -L$LP_VIEW/lib -llibpressio -Wl,-rpath,$LP_VIEW/lib
 *
 * Run:
 *   ./dump_opts sperr
 *   ./dump_opts sperr | tr ',' '\n' | grep -i chunk     <-- the answer
 */

#include <libpressio/libpressio.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void dump(const char *label, struct pressio_options *o)
{
    if (!o) { std::printf("=== %s: (null) ===\n\n", label); return; }
    char *s = pressio_options_to_string(o);
    std::printf("=== %s ===\n%s\n\n", label, s ? s : "(null)");
    std::free(s);
    pressio_options_free(o);
}

int main(int argc, char **argv)
{
    const char *id = (argc > 1) ? argv[1] : "sperr";

    struct pressio *library = pressio_instance();
    if (!library) { std::fprintf(stderr, "pressio_instance failed\n"); return 1; }

    struct pressio_compressor *c = pressio_get_compressor(library, id);
    if (!c) {
        /* Returns a comma-separated list in a library-owned buffer: no
         * arguments, and nothing to free. */
        const char *sup = pressio_supported_compressors();
        std::fprintf(stderr,
            "compressor '%s' is NOT available in this libpressio build: %s\n\n"
            "Compressors that ARE available:\n  %s\n",
            id, pressio_error_msg(library), sup ? sup : "(none reported)");
        pressio_release(library);
        return 2;
    }

    std::printf("compressor: %s\n\n", id);
    dump("OPTIONS (settable)", pressio_compressor_get_options(c));
    dump("CONFIGURATION (readonly / capabilities)",
                              pressio_compressor_get_configuration(c));
    dump("DOCUMENTATION",     pressio_compressor_get_documentation(c));

    pressio_compressor_release(c);
    pressio_release(library);

    std::printf(
      "--- what to look for ---\n"
      "  chunk dims : any key matching /chunk/. If present, set it to cover the\n"
      "               whole (sub)volume so SPERR emits ONE internal chunk.\n"
      "  error mode : the key selecting PWE vs PSNR vs fixed-rate, and whether\n"
      "               pressio:abs is among the settable options at all.\n"
      "  If NO chunk key exists, SPERR uses its 256^3 default and a byte prefix\n"
      "  of the stream is not independently decodable -- go to Plan B (the\n"
      "  256^3 crop) in bench_config_sperr.md.\n");
    return 0;
}
