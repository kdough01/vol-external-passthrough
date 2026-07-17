#include <libpressio/libpressio.h>
#include <cstdio>
#include <cstdlib>

int main(void) {
    struct pressio* lib = pressio_instance();

    struct pressio_compressor* c = pressio_get_compressor(lib, "cuszp");
    if (!c) {
        std::fprintf(stderr, "cuszp not available: %s\n", pressio_error_msg(lib));
        pressio_release(lib);
        return 1;
    }

    struct pressio_options* o = pressio_compressor_get_options(c);
    char* s = pressio_options_to_string(o);
    std::printf("%s\n", s ? s : "(null)");

    std::free(s);
    pressio_options_free(o);
    pressio_compressor_release(c);
    pressio_release(lib);
    return 0;
}