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
#include "vol_errors.h"

static void cuda_deleter(void* data, void* meta) {
    (void)meta;
    cudaFree(data);
}

/* Macro to check CUDA calls and push an HDF5 error on failure.
 * Jumps to `done` with ret_val = -1. */
#define CUDA_CHECK(call, msg)                                               \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,              \
                    vol_err_class, maj_compression, min_compress_failed,    \
                    msg ": %s", cudaGetErrorString(_e));                    \
            ret_val = -1;                                                   \
            goto done;                                                      \
        }                                                                   \
    } while(0)

extern "C" {

static int compressor_is_byte_stream(const char* id) {
    return strncmp(id, "nvcomp", 6) == 0;
}

static const char* gpu_stream_key(const char* id) {
    if (strncmp(id, "nvcomp", 6) == 0) return "nvcomp:stream";
    if (strncmp(id, "cuszp",  5) == 0) return "cuszp:stream";
    if (strncmp(id, "cusz",   4) == 0) return "cusz:stream";
    return NULL;
}

herr_t
H5VL_pass_through_ext_gpu_transfer_compress(gpu_vol_dataset_t* ds_ctx, const void* host_data, size_t nbytes)
{
    herr_t ret_val = 0;
    compression_ctx*  ctx = ds_ctx->comp_ctx;
    gpu_context_t*    gpu = ds_ctx->gpu_ctx;

    struct pressio_data* d_input  = NULL;
    struct pressio_data* d_output = NULL;
    struct pressio_options* stream_opts = NULL;

    int byte_stream = compressor_is_byte_stream(ctx->compressor_id);
    enum pressio_dtype in_dtype;
    size_t in_ndims;
    size_t* in_dims;
    size_t  byte_dims[1];

    if (byte_stream) {
        in_dtype   = pressio_byte_dtype;
        in_ndims   = 1;
        byte_dims[0] = nbytes;
        in_dims    = byte_dims;
    } else {
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "typed GPU compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        in_dtype = ctx->dtype;
        in_ndims = ctx->ndims;
        in_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("GPU COMPRESSION CALLED: id=%s nbytes=%zu dtype=%d ndims=%zu\n",
           ctx->compressor_id, nbytes, (int)in_dtype, in_ndims);
#endif

    /* Grow device input buffer if needed */
    if (nbytes > gpu->d_in_capacity) {
        CUDA_CHECK(cudaFree(gpu->d_in), "cudaFree of input buffer failed");
        CUDA_CHECK(cudaMalloc(&gpu->d_in, nbytes), "cudaMalloc for input buffer failed");
        gpu->d_in_capacity = nbytes;
    }

    CUDA_CHECK(cudaMemcpyAsync(gpu->d_in, host_data, nbytes,
                               cudaMemcpyHostToDevice, gpu->stream),
               "cudaMemcpyAsync host->device failed");

    d_input = pressio_data_new_nonowning(in_dtype, gpu->d_in, in_ndims, in_dims);
    d_output = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    {
        const char* skey = gpu_stream_key(ctx->compressor_id);
        if (skey) {
            stream_opts = pressio_options_new();
            pressio_options_set_userptr(stream_opts, skey, (void*)gpu->stream);
            pressio_compressor_set_options(ctx->compressor, stream_opts);
        }
        CUDA_CHECK(cudaStreamSynchronize(gpu->stream),
                   "cudaStreamSynchronize before compress failed");
    }

    if (pressio_compressor_compress(ctx->compressor, d_input, d_output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "GPU pressio_compressor_compress failed for '%s': %s",
                ctx->compressor_id,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }

    {
        size_t comp_size = 0;
        void* d_comp_ptr = pressio_data_ptr(d_output, &comp_size);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("GPU compress OK: id=%s comp_size=%zu\n", ctx->compressor_id, comp_size);
#endif
        if (comp_size == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "GPU compressor '%s' produced 0 bytes", ctx->compressor_id);
            ret_val = -1;
            goto done;
        }

        ctx->compressed_buf = malloc(comp_size);
        if (!ctx->compressed_buf) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory allocating %zu bytes for compressed output", comp_size);
            ret_val = -1;
            goto done;
        }

        CUDA_CHECK(cudaMemcpy(ctx->compressed_buf, d_comp_ptr, comp_size,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device->host for compressed data failed");

        ctx->compressed_chunk_size = comp_size;
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results =
            pressio_compressor_get_metrics_results(ctx->compressor);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] compress '%s':\n%s\n", ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (stream_opts) pressio_options_free(stream_opts);
    if (d_input)     pressio_data_free(d_input);
    if (d_output)    pressio_data_free(d_output);
    return ret_val;
}

static herr_t
H5VL_pass_through_ext_gpu_transfer_decompress(gpu_vol_dataset_t* ds_ctx, const void* compressed_host_data, size_t compressed_size, void* output_host_buf, size_t output_nbytes)
{
    herr_t ret_val = 0;
    compression_ctx*  ctx = ds_ctx->comp_ctx;
    gpu_context_t*    gpu = ds_ctx->gpu_ctx;

    void* d_comp = NULL;
    struct pressio_data*    d_input     = NULL;
    struct pressio_data*    d_output    = NULL;
    struct pressio_options* stream_opts = NULL;
    size_t comp_dims[1] = {compressed_size};

    int byte_stream = compressor_is_byte_stream(ctx->compressor_id);
    enum pressio_dtype out_dtype;
    size_t  out_ndims;
    size_t* out_dims;
    size_t  byte_dims[1];

    if (byte_stream) {
        out_dtype = pressio_byte_dtype;
        out_ndims = 1;
        byte_dims[0] = output_nbytes;
        out_dims = byte_dims;
    } else {
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "typed GPU compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        out_dtype = ctx->dtype;
        out_ndims = ctx->ndims;
        out_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("GPU DECOMPRESSION CALLED: id=%s compressed_size=%zu output_nbytes=%zu dtype=%d\n",
           ctx->compressor_id, compressed_size, output_nbytes, (int)out_dtype);
#endif

    CUDA_CHECK(cudaMalloc(&d_comp, compressed_size),
               "cudaMalloc for compressed input buffer failed");
    CUDA_CHECK(cudaMemcpy(d_comp, compressed_host_data, compressed_size,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host->device for compressed data failed");

    d_input = pressio_data_new_move(pressio_byte_dtype, d_comp, 1, comp_dims, cuda_deleter, NULL);
    d_comp  = NULL;

    d_output = pressio_data_new_empty(out_dtype, out_ndims, out_dims);

    {
        const char* skey = gpu_stream_key(ctx->compressor_id);
        if (skey) {
            stream_opts = pressio_options_new();
            pressio_options_set_userptr(stream_opts, skey, (void*)gpu->stream);
            pressio_compressor_set_options(ctx->compressor, stream_opts);
        }
    }

    if (pressio_compressor_decompress(ctx->compressor, d_input, d_output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "GPU pressio_compressor_decompress failed for '%s': %s",
                ctx->compressor_id,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }

    {
        size_t actual_bytes = 0;
        void* d_decomp_ptr = pressio_data_ptr(d_output, &actual_bytes);

        if (actual_bytes > output_nbytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "GPU decompress of '%s' produced %zu bytes, expected %zu",
                    ctx->compressor_id, actual_bytes, output_nbytes);
            ret_val = -1;
            goto done;
        }

        CUDA_CHECK(cudaMemcpy(output_host_buf, d_decomp_ptr, actual_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device->host for decompressed data failed");
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results =
            pressio_compressor_get_metrics_results(ctx->compressor);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] decompress '%s':\n%s\n", ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (stream_opts) pressio_options_free(stream_opts);
    if (d_input)     pressio_data_free(d_input);
    if (d_output)    pressio_data_free(d_output);
    if (d_comp)      cudaFree(d_comp);
    return ret_val;
}