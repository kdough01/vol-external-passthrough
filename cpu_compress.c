#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpressio/libpressio.h>
#include "H5VLpassthru_ext.h"
// #include <libpressio_ext/io/posix.h>

int cpu_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
    enum pressio_dtype dtype = pressio_byte_dtype;
    size_t flat_dim = nbytes;
    size_t *use_dims = &flat_dim;

    // new nonowning data
    // H5E
    struct pressio_data* input = pressio_data_new_copy(dtype, (void *)data, ndims, use_dims);
    struct pressio_data *compressed = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    if(pressio_compressor_compress(comp_ctx->compressor, input, compressed)) {
        fprintf(stderr, "%s\n", pressio_compressor_error_msg(comp_ctx->compressor));
        pressio_data_free(input);
        pressio_data_free(compressed);
        return pressio_compressor_error_code(comp_ctx->compressor);
    }

    size_t comp_size = 0;
    // store a pressio data object in the compression context
    void *comp_ptr = pressio_data_ptr(compressed, &comp_size);
    comp_ctx->compressed_buf = malloc(comp_size);
    memcpy(comp_ctx->compressed_buf, comp_ptr, comp_size);
    comp_ctx->compressed_chunk_size = comp_size;

    pressio_data_free(input);
    pressio_data_free(compressed);

    return 0;
}

herr_t
H5VL_pass_through_ext_cpu_transfer_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("CPU TRANSFORM CALLED\n");
    #endif

    cpu_compress(comp_ctx, data, nbytes);

    return 0;
}