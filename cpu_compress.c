#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpressio/libpressio.h>
#include "H5VLpassthru_ext.h"
// #include <libpressio_ext/io/posix.h>


herr_t
H5VL_pass_through_ext_cpu_transfer_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("------- CPU COMPRESSION CALLED\n");
    #endif

    struct pressio_data *input = pressio_data_new_nonowning(comp_ctx->dtype, (void *)data, comp_ctx->ndims, comp_ctx->dims);
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
H5VL_pass_through_ext_cpu_transfer_decompress(compression_ctx *comp_ctx,
    const void *compressed_data, size_t compressed_size,
    void *output_buf)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- CPU DECOMPRESSION CALLED\n");
#endif

    if (!comp_ctx || !comp_ctx->compressor) {
        fprintf(stderr, "Error: Invalid or uninitialized compression context in decompression.\n");
        return -1;
    }

    struct pressio_data *compressed = pressio_data_new_nonowning(
        pressio_byte_dtype, (void *)compressed_data,
        1, (size_t[]){compressed_size}
    );

    /* Safe explicit conversion from hsize_t (HDF5) to size_t (LibPressio) */
    size_t *lp_dims = (size_t *)malloc(comp_ctx->ndims * sizeof(size_t));
    for (int i = 0; i < comp_ctx->ndims; i++) {
        lp_dims[i] = (size_t)comp_ctx->dims[i];
    }

    /* Decompress directly into output buffer */
    struct pressio_data *decompressed = pressio_data_new_nonowning(
        comp_ctx->dtype, output_buf,
        comp_ctx->ndims, lp_dims
    );

    herr_t ret = 0;
    if (pressio_compressor_decompress(comp_ctx->compressor, compressed, decompressed)) {
        fprintf(stderr, "decompress: %s\n",
                pressio_compressor_error_msg(comp_ctx->compressor));
        ret = -1;
    }

    pressio_data_free(compressed);
    pressio_data_free(decompressed);
    free(lp_dims);
    
    return ret;
}