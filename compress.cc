#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

extern "C" {
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"
#include "vol_errors.h"
}

#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_manager.h>
#include "vol_timing_sink.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/* Helpers */
extern "C" int H5VL_pass_through_ext_buf_is_device(const void *p);

static int
vol_is_byte_stream(const char *id)
{
    return strncmp(id, "nvcomp", 6) == 0;
}

/* Codecs whose work runs on a CUDA stream and should be timed with events. */
static int
vol_is_gpu_codec(const char *id)
{
    return strncmp(id, "cuszp",  5) == 0 ||
           strncmp(id, "cusz",   4) == 0 ||
           strncmp(id, "nvcomp", 6) == 0;
}

static void
vol_make_host_resident(struct pressio_data *data)
{
    if (!data) return;

    if (strcmp(pressio_data_domain_id(data), "malloc") != 0) {
        pressio_data *d = data;
        *d = domain_manager().make_readable(
            libpressio::domain_plugins().build("malloc"), std::move(*d));
    }
}

static const char *
vol_ptr_domain(const void *p)
{
    return H5VL_pass_through_ext_buf_is_device(p) ? "cudamalloc" : "malloc";
}

extern "C" {

size_t
vol_logical_nbytes(const compression_ctx *ctx)
{
    size_t n = (size_t)pressio_dtype_size(ctx->dtype);
    for (size_t i = 0; i < ctx->ndims; i++)
        n *= ctx->dims[i];
    return n;
}

int
H5VL_pass_through_ext_compressor_available(const char *compressor_id)
{
    const char *list = pressio_supported_compressors();
    if (!list || !compressor_id)
        return 0;

    size_t idlen = strlen(compressor_id);
    const char *p = list;
    while (*p) {
        while (*p == ' ')
            p++;
        const char *start = p;
        while (*p && *p != ' ')
            p++;
        if ((size_t)(p - start) == idlen &&
            strncmp(start, compressor_id, idlen) == 0)
            return 1;
    }
    return 0;
}

/* Returns 1 if p is device- or managed-memory, 0 for host. Safe on any
 * pointer; clears CUDA's "not registered" error that host pointers produce. */
int
H5VL_pass_through_ext_buf_is_device(const void *p)
{
#ifdef USE_CUDA
    cudaPointerAttributes attr;
    if (p && cudaPointerGetAttributes(&attr, p) == cudaSuccess &&
        (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged))
        return 1;
    cudaGetLastError();
#else
    (void)p;
#endif
    return 0;
}

herr_t
H5VL_pass_through_ext_transfer_compress(compression_ctx *ctx, const void *data, size_t nbytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;

    enum pressio_dtype in_dtype;
    size_t  in_ndims;
    size_t *in_dims;
    size_t  byte_dims[1];

    if (vol_is_byte_stream(ctx->compressor_id)) {
        in_dtype     = pressio_byte_dtype;
        in_ndims     = 1;
        byte_dims[0] = nbytes;
        in_dims      = byte_dims;
    } else {
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            return -1;
        }

        /* libpressio sizes the clone/copy from the declared shape, not from
         * nbytes. If the recorded dims imply a different size than the buffer
         * HDF5 actually handed us, the copy runs off the end and segfaults.
         * Catch it here as a clean error instead. */
        size_t logical = vol_logical_nbytes(ctx);   /* dtype_size * prod(ctx->dims) */
        if (logical != nbytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "shape/size mismatch for '%s': recorded dims imply %zu bytes "
                    "but the write buffer is %zu bytes",
                    ctx->compressor_id, logical, nbytes);
            return -1;
        }

        in_dtype = ctx->dtype;
        in_ndims = ctx->ndims;
        in_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- TRANSFER COMPRESS: id=%s nbytes=%zu dtype=%d ndims=%zu\n",
           ctx->compressor_id, nbytes, (int)in_dtype, in_ndims);
#endif

    /* Always hand libpressio host memory; GPU compressors migrate it to the
     * device themselves via the domain manager. */
    input  = pressio_data_new_nonowning_domain(in_dtype, (void *)data, in_ndims, in_dims, "malloc");
    // output = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

    size_t max_comp_size = nbytes + 4096;
    size_t out_dims[1] = { max_comp_size };

    output = pressio_data_new_owning(pressio_byte_dtype, 1, out_dims);

    if (!output) {
        fprintf(stderr, "FATAL: Failed to allocate output buffer of size %zu\n", max_comp_size);
        abort();
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    {
        size_t shape_bytes = pressio_dtype_size(in_dtype);
        for (size_t i = 0; i < in_ndims; i++)
            shape_bytes *= in_dims[i];
        fprintf(stderr,
                "PRE-COMPRESS: data=%p nbytes=%zu ndims=%zu shape-bytes=%zu "
                "pressio_bytes=%zu has_data=%d  dims=[",
                data, nbytes, in_ndims, shape_bytes,
                pressio_data_get_bytes(input), pressio_data_has_data(input));
        for (size_t i = 0; i < in_ndims; i++)
            fprintf(stderr, "%zu%s", in_dims[i], (i + 1 < in_ndims) ? "," : "");
        fprintf(stderr, "]\n");
        fprintf(stderr, "  output=%p get_bytes=%zu capacity=%zu domain_id=%s\n",
                (void *)output, pressio_data_get_bytes(output),
                pressio_data_get_capacity_in_bytes(output),
                pressio_data_domain_id(output));
        const char* comp_id = pressio_compressor_get_name(ctx->compressor);
        fprintf(stderr, "DEBUG: Current compressor ID is: %s\n", comp_id);
        fflush(stderr);
    }
#endif

    if (!ctx) {
        fprintf(stderr, "CRITICAL CRASH AVOIDED: ctx is NULL!\n");
        abort();
    }
    if (!ctx->compressor) {
        fprintf(stderr, "CRITICAL CRASH AVOIDED: ctx->compressor is NULL!\n");
        abort();
    }

    size_t out_cap = pressio_data_get_capacity_in_bytes(output);
    fprintf(stderr, "DEBUG: Right before compress. Compressor=%p, Output Capacity=%zu bytes\n",
            (void*)ctx->compressor, out_cap);

    if (out_cap == 0) {
        fprintf(stderr, "WARNING: Passing a 0-capacity output buffer to the compressor.\n");
    }

    ctx->compress_ms = 0.0;   /* 0 => caller uses its wall clock (CPU codecs) */

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default stream */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _cerr = pressio_compressor_compress(ctx->compressor, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);           /* wait for H2D + kernel */
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms = (double)_ms;       /* device compress time */
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_cerr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "pressio_compressor_compress failed for '%s': %s",
                    ctx->compressor_id,
                    pressio_compressor_error_msg(ctx->compressor));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_compress(ctx->compressor, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio_compressor_compress failed for '%s': %s",
                ctx->compressor_id,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
#endif

    /* The result may be device-resident (GPU compressors). */
    vol_make_host_resident(output);

    {
        size_t comp_size = 0;
        void  *comp_ptr  = pressio_data_ptr(output, &comp_size);

        if (comp_size == 0 || comp_ptr == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "compressor '%s' produced 0 bytes", ctx->compressor_id);
            ret_val = -1;
            goto done;
        }

        ctx->compressed_buf = malloc(comp_size);
        if (!ctx->compressed_buf) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory allocating %zu bytes for compressed output",
                    comp_size);
            ret_val = -1;
            goto done;
        }

        memcpy(ctx->compressed_buf, comp_ptr, comp_size);
        ctx->compressed_chunk_size = comp_size;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("TRANSFER COMPRESS OK: id=%s comp_size=%zu original_nbytes=%zu\n",
               ctx->compressor_id, comp_size, nbytes);
#endif
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
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_transfer_decompress(compression_ctx *ctx,
                                          const void *compressed_data,
                                          size_t compressed_size,
                                          void *output_buf)
{
    if (!ctx || !ctx->compressor) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid or uninitialized compression context");
        return -1;
    }

    /* noop: libpressio's noop rejects a typed output buffer, so copy directly. */
    if (strcmp(ctx->compressor_id, "noop") == 0) {
        size_t logical = vol_logical_nbytes(ctx);
        size_t n = compressed_size < logical ? compressed_size : logical;
        memcpy(output_buf, compressed_data, n);
        return 0;
    }

    if (ctx->ndims == 0 || ctx->dims == NULL) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "compressor '%s' needs dtype/shape, but none recorded",
                ctx->compressor_id);
        return -1;
    }

    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    size_t comp_dims[1] = { compressed_size };

    const size_t out_nbytes = vol_logical_nbytes(ctx);

    enum pressio_dtype out_dtype;
    size_t  out_ndims;
    size_t *out_dims;
    size_t  byte_dims[1];

    if (vol_is_byte_stream(ctx->compressor_id)) {
        out_dtype    = pressio_byte_dtype;
        out_ndims    = 1;
        byte_dims[0] = out_nbytes;
        out_dims     = byte_dims;
    } else {
        out_dtype = ctx->dtype;
        out_ndims = ctx->ndims;
        out_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- TRANSFER DECOMPRESS: id=%s compressed_size=%zu "
           "out_nbytes=%zu dtype=%d ndims=%zu\n",
           ctx->compressor_id, compressed_size, out_nbytes,
           (int)out_dtype, out_ndims);
#endif

    input = pressio_data_new_nonowning_domain(pressio_byte_dtype,
                                              (void *)compressed_data,
                                              1, comp_dims, "malloc");

    /* Give libpressio an OWNING output, exactly like the compress path.
     * GPU compressors (cuszp) decompress into device memory; if we pre-bind
     * the caller's buffer as nonowning host, make_writeable can't relocate it
     * to the device, the result never comes back, and output_buf is left
     * uninitialized. Let libpressio place the output wherever it needs, pull
     * it home, then copy into the caller's buffer. */
    output = pressio_data_new_owning(out_dtype, out_ndims, out_dims);
    if (!output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating %zu-byte decompress output for '%s'",
                out_nbytes, ctx->compressor_id);
        ret_val = -1;
        goto done;
    }

    ctx->compress_ms = 0.0;   /* reused as device codec time for this leg */

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default stream */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _derr = pressio_compressor_decompress(ctx->compressor, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms = (double)_ms;       /* device decompress time */
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_derr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio_compressor_decompress failed for '%s': %s",
                    ctx->compressor_id,
                    pressio_compressor_error_msg(ctx->compressor));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_decompress(ctx->compressor, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio_compressor_decompress failed for '%s': %s",
                ctx->compressor_id,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
#endif
    fprintf(stderr, "DECOMP domain id = %s\n", pressio_data_domain_id(output));

    /* device -> host for GPU compressors, no-op for CPU. */
    vol_make_host_resident(output);

    {
        size_t actual_bytes = 0;
        void  *out_ptr = pressio_data_ptr(output, &actual_bytes);

        if (!out_ptr || actual_bytes == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "decompress of '%s' produced no host-resident output",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        if (actual_bytes > out_nbytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "decompress of '%s' produced %zu bytes, expected %zu",
                    ctx->compressor_id, actual_bytes, out_nbytes);
            ret_val = -1;
            goto done;
        }

        memcpy(output_buf, out_ptr, actual_bytes);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("TRANSFER DECOMPRESS OK: id=%s actual_bytes=%zu\n",
               ctx->compressor_id, actual_bytes);
#endif
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
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_transfer_compress_chunk(compression_ctx *ctx,
                                              const void *data, size_t nbytes,
                                              void **out_cbuf, uint64_t *out_csize)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype in_dtype;
    size_t chunk_dims[1];
    size_t out_dims[1];

    if (!ctx || !ctx->compressor || !out_cbuf || !out_csize || !data) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "invalid arguments to compress_chunk");
        return -1;
    }
    *out_cbuf  = NULL;
    *out_csize = 0;

    if (vol_is_byte_stream(ctx->compressor_id)) {
        in_dtype      = pressio_byte_dtype;
        chunk_dims[0] = nbytes;
    } else {
        const size_t dsize = pressio_dtype_size(ctx->dtype);
        if (nbytes == 0 || nbytes % dsize != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "chunk of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", nbytes, dsize, ctx->compressor_id);
            return -1;
        }
        in_dtype      = ctx->dtype;
        chunk_dims[0] = nbytes / dsize;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- COMPRESS CHUNK: id=%s nbytes=%zu nelem=%zu dtype=%d\n",
           ctx->compressor_id, nbytes, chunk_dims[0], (int)in_dtype);
#endif

    /* Host memory in; GPU compressors migrate via the domain manager. */
    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              1, chunk_dims,
                                              vol_ptr_domain(data));

    out_dims[0] = nbytes + 4096;
    output = pressio_data_new_owning(pressio_byte_dtype, 1, out_dims);

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte chunk", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default stream */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _cerr = pressio_compressor_compress(ctx->compressor, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms += (double)_ms;      /* ACCUMULATE across chunks */
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_cerr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "pressio_compressor_compress failed for '%s' (chunk of %zu B): %s",
                    ctx->compressor_id, nbytes,
                    pressio_compressor_error_msg(ctx->compressor));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_compress(ctx->compressor, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio_compressor_compress failed for '%s' (chunk of %zu B): %s",
                ctx->compressor_id, nbytes,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
#endif

    /* The result may be device-resident (GPU compressors). */
    vol_make_host_resident(output);

    {
        size_t comp_size = 0;
        void  *comp_ptr  = pressio_data_ptr(output, &comp_size);

        if (comp_size == 0 || comp_ptr == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "compressor '%s' produced 0 bytes for a %zu byte chunk",
                    ctx->compressor_id, nbytes);
            ret_val = -1;
            goto done;
        }

        void *cb = malloc(comp_size);
        if (!cb) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory allocating %zu bytes for compressed chunk",
                    comp_size);
            ret_val = -1;
            goto done;
        }

        memcpy(cb, comp_ptr, comp_size);
        *out_cbuf  = cb;
        *out_csize = (uint64_t)comp_size;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("COMPRESS CHUNK OK: id=%s comp_size=%zu original_nbytes=%zu\n",
               ctx->compressor_id, comp_size, nbytes);
#endif
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results =
            pressio_compressor_get_metrics_results(ctx->compressor);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] compress chunk '%s':\n%s\n", ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_transfer_decompress_chunk(compression_ctx *ctx,
                                                const void *cbuf, size_t csize,
                                                void *out, size_t out_bytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype out_dtype;
    size_t comp_dims[1];
    size_t out_dims[1];

    if (!ctx || !ctx->compressor || !cbuf || !out) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid arguments to decompress_chunk");
        return -1;
    }

    /* noop: libpressio's noop rejects a typed output buffer, copy directly. */
    if (strcmp(ctx->compressor_id, "noop") == 0) {
        size_t n = csize < out_bytes ? csize : out_bytes;
        memcpy(out, cbuf, n);
        return 0;
    }

    if (vol_is_byte_stream(ctx->compressor_id)) {
        out_dtype   = pressio_byte_dtype;
        out_dims[0] = out_bytes;
    } else {
        const size_t dsize = pressio_dtype_size(ctx->dtype);
        if (out_bytes == 0 || out_bytes % dsize != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "chunk output of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", out_bytes, dsize, ctx->compressor_id);
            return -1;
        }
        out_dtype   = ctx->dtype;
        out_dims[0] = out_bytes / dsize;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS CHUNK: id=%s csize=%zu out_bytes=%zu nelem=%zu\n",
           ctx->compressor_id, csize, out_bytes, out_dims[0]);
#endif

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");

    output = pressio_data_new_owning(out_dtype, 1, out_dims);
    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating %zu-byte chunk decompress output for '%s'",
                out_bytes, ctx->compressor_id);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default stream */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _derr = pressio_compressor_decompress(ctx->compressor, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms += (double)_ms;      /* ACCUMULATE across chunks */
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_derr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio_compressor_decompress failed for '%s' (chunk of %zu B): %s",
                    ctx->compressor_id, csize,
                    pressio_compressor_error_msg(ctx->compressor));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_decompress(ctx->compressor, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio_compressor_decompress failed for '%s' (chunk of %zu B): %s",
                ctx->compressor_id, csize,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
#endif

    /* device -> host for GPU compressors, no-op for CPU. */
    vol_make_host_resident(output);

    {
        size_t actual_bytes = 0;
        void  *out_ptr = pressio_data_ptr(output, &actual_bytes);

        if (!out_ptr || actual_bytes == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "chunk decompress of '%s' produced no host-resident output",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        if (actual_bytes != out_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "chunk decompress of '%s' produced %zu bytes, expected %zu",
                    ctx->compressor_id, actual_bytes, out_bytes);
            ret_val = -1;
            goto done;
        }

        memcpy(out, out_ptr, out_bytes);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("DECOMPRESS CHUNK OK: id=%s actual_bytes=%zu\n",
               ctx->compressor_id, actual_bytes);
#endif
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}
} /* extern C */