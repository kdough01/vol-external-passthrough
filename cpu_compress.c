#include <stdio.h>
#include <stdlib.h>
#include <libpressio.h>
#include <libpressio_ext/io/posix.h>






int cpu_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
    pressio_dtype = pressio_byte_dtype;
    size_t ndims = 1;
    size_t flast_dim = nbytes;
    size_t *use_dims = &flat_dim;

    struct pressio_data* input = pressio_data_new_copy(dtype, data, ndims, use_dims);
    struct pressio_data *compressed = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    if(pressio_compressor_compress(comp, input_data, compressed)) {
        fprintf(stderr, "%s\n", pressio_compressor_error_msg(comp));
        pressio_data_free(input);
        pressio_data_free(compressed);
        return pressio_compressor_error_code(comp);
    }

    size_t comp_size = 0;
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
        printf("CPU TRANSFORM CALLED: nelem=%zu dtype=%d\n", nelem, dtype);
    #endif

    void *d_dset;
    H5T_class_t cls = H5Tget_class(dtype);
    size_t size = H5Tget_size(dtype);
    size_t bytes = nelem * H5Tget_size(dtype);

    cpu_compress(ctx);

    return 0;
}