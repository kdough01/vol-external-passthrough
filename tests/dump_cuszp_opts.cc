/* dump_opts.cpp -- print every option a libpressio compressor exposes.
 *
 * Generic version of your dump_cuszp_opts. One decisive question here:
 * does the sperr plugin let us set SPERR's internal chunk dimensions?
 * If it does, progressive truncation is safe. If it does not, SPERR's
 * default 256^3 blocking splits every dataset you have into multiple
 * internal chunks and a prefix of the bitstream is not decodable.
 *
 * Build (same flags as your other bench tools):
 *   g++ -O2 -std=c++17 dump_opts.cpp -o dump_opts \
 *       $(pkg-config --cflags --libs libpressio)
 * or, matching your CMake setup:
 *   g++ -O2 -std=c++17 dump_opts.cpp -o dump_opts \
 *       -I$LP_VIEW/include -L$LP_VIEW/lib64 -L$LP_VIEW/lib \
 *       -llibpressio -Wl,-rpath,$LP_VIEW/lib64 -Wl,-rpath,$LP_VIEW/lib
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
        std::fprintf(stderr,
            "compressor '%s' is NOT available in this libpressio build: %s\n"
            "\nCompressors that ARE available:\n", id, pressio_error_msg(library));
        struct pressio_options *sup = pressio_supported_compressors(library);
        if (sup) {
            char *s = pressio_options_to_string(sup);
            std::fprintf(stderr, "%s\n", s ? s : "(none)");
            std::free(s);
            pressio_options_free(sup);
        }
        pressio_release(library);
        return 2;
    }

    std::printf("compressor: %s\n\n", id);
    dump("OPTIONS (settable)",  pressio_compressor_get_options(c));
    dump("CONFIGURATION (readonly / capabilities)",
                               pressio_compressor_get_configuration(c));
    dump("DOCUMENTATION",      pressio_compressor_get_documentation(c));

    pressio_compressor_release(c);
    pressio_release(library);

    std::printf(
      "--- what to look for ---\n"
      "  chunk dims : any key matching /chunk/ -- if present, set it to cover\n"
      "               the whole (sub)volume so SPERR emits ONE internal chunk.\n"
      "  error mode : the key that selects PWE vs PSNR vs fixed-rate.\n"
      "  If NO chunk key exists, SPERR uses its 256^3 default and a byte prefix\n"
      "  of the stream is not independently decodable -- see Plan B in\n"
      "  bench_config_sperr.md.\n");
    return 0;
}