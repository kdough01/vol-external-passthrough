#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"

static void cuda_deleter(void* data, void* meta) {
    (void)meta;
    cudaFree(data);
}

extern "C" {

herr_t
H5VL_pass_through_ext_gpu_transfer_compress(gpu_vol_dataset_t* ds_ctx, const void* host_data, size_t nbytes)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("GPU COMPRESSION CALLED: nbytes=%zu\n", nbytes);
#endif

    compression_ctx*  ctx = ds_ctx->comp_ctx;
    gpu_context_t*    gpu = ds_ctx->gpu_ctx;

    if (nbytes > gpu->d_in_capacity) {
        cudaFree(gpu->d_in);
        cudaMalloc(&gpu->d_in, nbytes);
        gpu->d_in_capacity = nbytes;
    }
    cudaMemcpyAsync(gpu->d_in, host_data, nbytes, cudaMemcpyHostToDevice, gpu->stream);

    size_t dims[1] = {nbytes};
    struct pressio_data* d_input  = pressio_data_new_nonowning(pressio_byte_dtype, gpu->d_in, 1, dims);
    struct pressio_data* d_output = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    struct pressio_options* stream_opts = pressio_options_new();
    pressio_options_set_userptr(stream_opts, "nvcomp:stream", (void*)gpu->stream);
    pressio_compressor_set_options(ctx->compressor, stream_opts);
    pressio_options_free(stream_opts);

    if (pressio_compressor_compress(ctx->compressor, d_input, d_output)) {
        fprintf(stderr, "GPU compress error: %s\n", pressio_compressor_error_msg(ctx->compressor));
        pressio_data_free(d_input);
        pressio_data_free(d_output);
        return -1;
    }

    size_t comp_size = 0;
    void* d_comp_ptr = pressio_data_ptr(d_output, &comp_size);

    ctx->compressed_buf        = malloc(comp_size);
    cudaMemcpy(ctx->compressed_buf, d_comp_ptr, comp_size, cudaMemcpyDeviceToHost);
    ctx->compressed_chunk_size = comp_size;

    pressio_data_free(d_input);
    pressio_data_free(d_output);
    return 0;
}

static herr_t
H5VL_pass_through_ext_gpu_transfer_decompress(gpu_vol_dataset_t* ds_ctx, const void* compressed_host_data, size_t compressed_size, void* output_host_buf, size_t output_nbytes)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("GPU DECOMPRESSION CALLED: compressed_size=%zu output_nbytes=%zu\n", compressed_size, output_nbytes);
#endif

    compression_ctx*  ctx = ds_ctx->comp_ctx;
    gpu_context_t*    gpu = ds_ctx->gpu_ctx;

    void* d_comp;
    cudaMalloc(&d_comp, compressed_size);
    cudaMemcpy(d_comp, compressed_host_data, compressed_size, cudaMemcpyHostToDevice);

    size_t comp_dims[1] = {compressed_size};
    struct pressio_data* d_input = pressio_data_new_move(pressio_byte_dtype, d_comp, 1, comp_dims, cuda_deleter, NULL);

    size_t out_dims[1] = {output_nbytes};
    struct pressio_data* d_output = pressio_data_new_empty(pressio_byte_dtype, 1, out_dims);

    struct pressio_options* stream_opts = pressio_options_new();
    pressio_options_set_userptr(stream_opts, "nvcomp:stream", (void*)gpu->stream);
    pressio_compressor_set_options(ctx->compressor, stream_opts);
    pressio_options_free(stream_opts);

    if (pressio_compressor_decompress(ctx->compressor, d_input, d_output)) {
        fprintf(stderr, "GPU decompress error: %s\n", pressio_compressor_error_msg(ctx->compressor));
        pressio_data_free(d_input);
        pressio_data_free(d_output);
        return -1;
    }

    size_t actual_bytes = 0;
    void*  d_decomp_ptr = pressio_data_ptr(d_output, &actual_bytes);
    cudaMemcpy(output_host_buf, d_decomp_ptr, actual_bytes, cudaMemcpyDeviceToHost);

    pressio_data_free(d_input);
    pressio_data_free(d_output);
    return 0;
}

}