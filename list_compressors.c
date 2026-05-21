#include <stdio.h>
#include <libpressio.h>

int main() {
    printf("%s\n", pressio_supported_compressors());
}