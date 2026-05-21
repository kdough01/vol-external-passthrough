#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpressio/libpressio.h>
#include "H5VLpassthru_ext.h"
// #include <libpressio_ext/io/posix.h>

int cpu_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
    // new nonowning data
    // H5E
    printf("comp_ctx=%p\n", (void*)comp_ctx);
    printf("data=%p\n", data);
    printf("ndims=%zu\n", comp_ctx->ndims);
    for (size_t i = 0; i < comp_ctx->ndims; i++) {
        printf("dims[%zu]=%zu\n", i, comp_ctx->dims[i]);
    }
    printf("dtype=%d\n", comp_ctx->dtype);
    printf("nbytes=%zu\n", nbytes);
    struct pressio_data *input = pressio_data_new_nonowning(comp_ctx->dtype, (void *)data, comp_ctx->ndims, comp_ctx->dims);
    printf("two\n");
    struct pressio_data *compressed = pressio_data_new_empty(comp_ctx->dtype, 0, NULL);

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