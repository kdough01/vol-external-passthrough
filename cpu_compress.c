#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpressio/libpressio.h>
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"
#include "vol_errors.h"

herr_t
H5VL_pass_through_ext_cpu_transfer_compress(compression_ctx *comp_ctx, const void *data, size_t nbytes)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- CPU COMPRESSION CALLED\n");
#endif

    herr_t ret_val = 0;
    struct pressio_data *input      = NULL;
    struct pressio_data *compressed = NULL;

    input      = pressio_data_new_nonowning(comp_ctx->dtype, (void *)data, comp_ctx->ndims, comp_ctx->dims);
    compressed = pressio_data_new_empty(comp_ctx->dtype, 0, NULL);

    if (pressio_compressor_compress(comp_ctx->compressor, input, compressed)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio_compressor_compress failed for '%s': %s",
                comp_ctx->compressor_id,
                pressio_compressor_error_msg(comp_ctx->compressor));
        ret_val = -1;
        goto done;
    }

    {
        size_t comp_size = 0;
        void *comp_ptr = pressio_data_ptr(compressed, &comp_size);
        comp_ctx->compressed_buf = malloc(comp_size);
        memcpy(comp_ctx->compressed_buf, comp_ptr, comp_size);
        comp_ctx->compressed_chunk_size = comp_size;

        printf("DEBUG compress: comp_size=%zu original_nbytes=%zu dtype=%d\n",
               comp_size, nbytes, (int)comp_ctx->dtype);
    }

done:
    pressio_data_free(input);
    pressio_data_free(compressed);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_cpu_transfer_decompress(compression_ctx *comp_ctx,
    const void *compressed_data, size_t compressed_size,
    void *output_buf)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- CPU DECOMPRESSION CALLED\n");
#endif

    printf("DEBUG decompress entry: ndims=%zu dims=[", comp_ctx->ndims);
    for (size_t i = 0; i < comp_ctx->ndims; i++)
        printf("%zu%s", comp_ctx->dims[i], i+1 < comp_ctx->ndims ? "," : "");
    printf("] dtype=%d compressed_size=%zu output_buf=%p\n",
           (int)comp_ctx->dtype, compressed_size, output_buf);
    fflush(stdout);

    if (!comp_ctx || !comp_ctx->compressor) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid or uninitialized compression context");
        return -1;
    }

    if (strcmp(comp_ctx->compressor_id, "noop") == 0) {
        memcpy(output_buf, compressed_data, compressed_size);
        return 0;
    }

    herr_t ret_val = 0;
    size_t *lp_dims             = NULL;
    struct pressio_data *compressed   = NULL;
    struct pressio_data *decompressed = NULL;

    compressed = pressio_data_new_nonowning(
        pressio_byte_dtype, (void *)compressed_data,
        1, (size_t[]){compressed_size}
    );

    lp_dims = (size_t *)malloc(comp_ctx->ndims * sizeof(size_t));
    for (size_t i = 0; i < comp_ctx->ndims; i++)
        lp_dims[i] = comp_ctx->dims[i];

    decompressed = pressio_data_new_nonowning(
        comp_ctx->dtype, output_buf,
        comp_ctx->ndims, lp_dims
    );

    if (pressio_compressor_decompress(comp_ctx->compressor, compressed, decompressed)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio_compressor_decompress failed for '%s': %s",
                comp_ctx->compressor_id,
                pressio_compressor_error_msg(comp_ctx->compressor));
        ret_val = -1;
        goto done;
    }

    {
        size_t out_size = 0;
        void *out_ptr = pressio_data_ptr(decompressed, &out_size);
        printf("DEBUG decompress success: out_ptr=%p out_size=%zu first_float=%e\n",
               out_ptr, out_size, ((float*)out_ptr)[0]);
        printf("DEBUG output_buf=%p first_float=%e\n",
               output_buf, ((float*)output_buf)[0]);
    }

done:
    pressio_data_free(compressed);
    pressio_data_free(decompressed);
    free(lp_dims);
    return ret_val;
}