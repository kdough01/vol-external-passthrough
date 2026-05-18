#include <stdio.h>
#include <stdlib.h>
#include <libpressio.h>
#include <libpressio_ext/io/posix.h>

/*
Metadata for compressor:
- compressor used
error
*/
int cpu_compress(struct compression_ctx *ctx)
{

}

herr_t
H5VL_pass_through_ext_cpu_transfer_compress()
{
    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("CPU TRANSFORM CALLED: nelem=%zu dtype=%d\n", nelem, dtype);
    #endif

    void *d_dset;
    H5T_class_t cls = H5Tget_class(dtype);
    size_t size = H5Tget_size(dtype);
    size_t bytes = nelem * H5Tget_size(dtype);

    cpu_compress(ctx);

    return 0;
}