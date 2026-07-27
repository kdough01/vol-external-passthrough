#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <strings.h>

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
extern "C" int H5VL_pass_through_ext_compressor_available(const char *compressor_id);

/* Codecs that receive a 1-D flattened byte stream instead of a typed,
 * shaped view. Lossless byte-oriented codecs only — shape-aware codecs
 * (roibin, sz3, zfp, cusz) must keep their dims. */
static int
vol_is_byte_stream(const char *id)
{
    return strncmp(id, "nvcomp", 6) == 0 ||
           strcmp(id, "zstd") == 0;
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

static struct pressio_compressor *
vol_get_chunk_wrapper(compression_ctx *ctx, uint64_t chunk_elems)
{
    if (ctx->chunk_wrapper && ctx->chunk_wrapper_elems == chunk_elems)
        return ctx->chunk_wrapper;

    if (ctx->chunk_wrapper) {
        pressio_compressor_release(ctx->chunk_wrapper);
        ctx->chunk_wrapper       = NULL;
        ctx->chunk_wrapper_elems = 0;
    }

    if (!H5VL_pass_through_ext_compressor_available("chunking") ||
        !H5VL_pass_through_ext_compressor_available("many_independent")) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compressor_unavail,
                "libpressio 'chunking'+'many_independent' meta-compressors are "
                "not available in this build; use chunking_mode 'none' or 'vol'");
        return NULL;
    }

    struct pressio *lib = pressio_instance();
    struct pressio_compressor *w = lib ? pressio_get_compressor(lib, "chunking") : NULL;
    if (lib) pressio_release(lib);
    if (!w) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compressor_unavail,
                "failed to instantiate libpressio 'chunking' meta-compressor");
        return NULL;
    }

    /* Same recipe as the old vol_make_chunking_compressor: start from the
     * child codec's full option set (carries tolerances, userptr stream,
     * etc.), then splice in the chunking -> many_independent -> codec chain
     * and the chunk size, applied in a single set_options call. */
    {
        struct pressio_options *opts =
            pressio_compressor_get_options(ctx->compressor);
        if (!opts)
            opts = pressio_options_new();

        pressio_options_set_string(opts, "chunking:compressor",
                                   "many_independent");
        pressio_options_set_string(opts, "many_independent:compressor",
                                   ctx->compressor_id);

        {
            size_t one = 1;
            size_t csz_bytes = 0;
            struct pressio_data *csz =
                pressio_data_new_owning(pressio_uint64_dtype, 1, &one);
            ((uint64_t *)pressio_data_ptr(csz, &csz_bytes))[0] = chunk_elems;
            pressio_options_set_data(opts, "chunking:size", csz);
            pressio_data_free(csz);
        }

        const char *nt = getenv("VOL_COMP_PRESSIO_NTHREADS");
        if (nt && *nt) {
            unsigned long v = strtoul(nt, NULL, 10);
            if (v > 0)
                pressio_options_set_uinteger(opts, "chunking:nthreads",
                                             (unsigned)v);
        }

        int err = pressio_compressor_set_options(w, opts);
        pressio_options_free(opts);
        if (err) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compressor_unavail,
                    "failed to configure chunking/many_independent wrapper for "
                    "'%s' (chunk_elems=%llu): %s",
                    ctx->compressor_id, (unsigned long long)chunk_elems,
                    pressio_compressor_error_msg(w));
            pressio_compressor_release(w);
            return NULL;
        }
    }

#ifdef USE_CUDA
    /* Re-apply the CUDA stream through the wrapper (the child opts replayed
     * above should already cover this; cheap belt-and-braces). */
    if (ctx->stream) {
        struct pressio_options *s = pressio_options_new();
        char skey[128];
        snprintf(skey, sizeof(skey), "%s:cuda_stream", ctx->compressor_id);
        pressio_options_set_userptr(s, skey, ctx->stream);
        (void)pressio_compressor_set_options(w, s);
        pressio_options_free(s);
    }
#endif

    ctx->chunk_wrapper       = w;
    ctx->chunk_wrapper_elems = chunk_elems;
    return w;
}

static const char *
vol_ptr_domain(const void *p)
{
    return H5VL_pass_through_ext_buf_is_device(p) ? "cudamalloc" : "malloc";
}

#ifdef USE_CUDA
/* Hand the codec our CUDA stream (previously done inside the chunking
 * pipeline builder; now applied directly to ctx->compressor). Idempotent. */
static void
vol_set_cuda_stream(compression_ctx *ctx)
{
    if (!ctx->stream) return;
    struct pressio_options *sopt = pressio_options_new();
    char skey[128];
    snprintf(skey, sizeof(skey), "%s:cuda_stream", ctx->compressor_id);
    pressio_options_set_userptr(sopt, skey, ctx->stream);
    (void)pressio_compressor_set_options(ctx->compressor, sopt);
    pressio_options_free(sopt);
}
#endif

extern "C" {

size_t
vol_logical_nbytes(const compression_ctx *ctx)
{
    size_t n = (size_t)pressio_dtype_size(ctx->dtype);
    for (size_t i = 0; i < ctx->ndims; i++)
        n *= ctx->dims[i];
    return n;
}

/*
 * Path selection knob.
 *
 * Returns the VOL-level chunk size in bytes, or 0 when VOL_COMP_CHUNK_MB is
 * unset/invalid. 0 means "native compressor chunking": the whole dataset is
 * handed to the compressor in a single compress call and the codec does its
 * own internal blocking (SZ3 blocks, ZFP 4^d blocks, nvcomp batching, ...).
 *
 * Setting VOL_COMP_CHUNK_MB=<n> switches to VOL-level chunking: the VOL
 * splits the buffer into <n>-MB pieces and compresses each independently
 * (e.g. VOL_COMP_CHUNK_MB=1024 reproduces the old 1 GB behavior).
 */
size_t
vol_comp_chunk_bytes(size_t dsize)
{
    const char *env = getenv("VOL_COMP_CHUNK_MB");
    if (!env || !*env)
        return 0;                       /* default: native compressor chunking */

    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || v == 0)
        return 0;

    if (dsize == 0) dsize = 1;
    size_t bytes = (size_t)v << 20;
    bytes -= bytes % dsize;             /* whole elements only */
    if (bytes < dsize) bytes = dsize;
    return bytes;
}

/* Native compressor chunking */
static size_t
vol_native_chunk_elems(compression_ctx *ctx, size_t dsize)
{
    (void)ctx;
    size_t mb = 1024;
    const char *env = getenv("VOL_COMP_CHUNK_MB");
    if (env && *env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0)
            mb = (size_t)v;
    }
    if (dsize == 0) dsize = 1;
    size_t bytes = mb << 20;
    bytes -= bytes % dsize;
    if (bytes < dsize) bytes = dsize;
    size_t elems = bytes / dsize;
    return elems ? elems : 1;
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

/* ========================================================================
 * NATIVE PATH (default): one compress/decompress call over the whole
 * dataset, straight into ctx->compressor. No chunking wrapper of any kind;
 * whatever chunking happens is the codec's own internal blocking.
 * ======================================================================== */

herr_t
H5VL_pass_through_ext_compress_native(compression_ctx *ctx,
                                      const void *data, size_t nbytes,
                                      void **out_cbuf, uint64_t *out_csize)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype in_dtype;
    size_t  in_ndims;
    size_t *in_dims;
    size_t  byte_dims[1];
    size_t  out_dims[1];

    if (!ctx || !ctx->compressor || !out_cbuf || !out_csize || !data) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "invalid arguments to compress_native");
        return -1;
    }
    *out_cbuf  = NULL;
    *out_csize = 0;

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

        /* libpressio sizes the copy from the declared shape, not nbytes.
         * A mismatch would run off the end of the buffer; fail cleanly. */
        size_t logical = vol_logical_nbytes(ctx);
        if (logical != nbytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "shape/size mismatch for '%s': recorded dims imply %zu bytes "
                    "but the write buffer is %zu bytes",
                    ctx->compressor_id, logical, nbytes);
            return -1;
        }

        /* Full multi-D shape: lets shape-aware codecs (sz3, zfp, cusz)
         * exploit dimensionality instead of a flattened 1-D view. */
        in_dtype = ctx->dtype;
        in_ndims = ctx->ndims;
        in_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- COMPRESS NATIVE: id=%s nbytes=%zu ndims=%zu dtype=%d "
           "(single call, codec-internal chunking)\n",
           ctx->compressor_id, nbytes, in_ndims, (int)in_dtype);
#endif

    /* Device buffers pass through as-is; GPU codecs migrate host input via
     * the domain manager themselves. */
    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              in_ndims, in_dims,
                                              vol_ptr_domain(data));

    out_dims[0] = nbytes + 4096;
    output = pressio_data_new_owning(pressio_byte_dtype, 1, out_dims);

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte dataset", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    vol_set_cuda_stream(ctx);
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
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
            ctx->compress_ms += (double)_ms;
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_cerr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "native compress failed for '%s' (%zu B): %s",
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
                "native compress failed for '%s' (%zu B): %s",
                ctx->compressor_id, nbytes,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
#endif

    /* Result may be device-resident for GPU compressors. */
    vol_make_host_resident(output);

    {
        size_t comp_size = 0;
        void  *comp_ptr  = pressio_data_ptr(output, &comp_size);

        if (comp_size == 0 || comp_ptr == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "native compress of '%s' produced 0 bytes for %zu B",
                    ctx->compressor_id, nbytes);
            ret_val = -1;
            goto done;
        }

        void *cb = malloc(comp_size);
        if (!cb) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory copying %zu compressed bytes", comp_size);
            ret_val = -1;
            goto done;
        }

        memcpy(cb, comp_ptr, comp_size);
        *out_cbuf  = cb;
        *out_csize = (uint64_t)comp_size;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("COMPRESS NATIVE OK: id=%s comp_size=%zu original_nbytes=%zu\n",
               ctx->compressor_id, comp_size, nbytes);
#endif
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results =
            pressio_compressor_get_metrics_results(ctx->compressor);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] compress native '%s':\n%s\n", ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_decompress_native(compression_ctx *ctx,
                                        const void *cbuf, size_t csize,
                                        void *out, size_t out_bytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype out_dtype;
    size_t  out_ndims;
    size_t *out_dims;
    size_t  byte_dims[1];
    size_t  comp_dims[1];

    if (!ctx || !ctx->compressor || !cbuf || !out) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid arguments to decompress_native");
        return -1;
    }

    /* noop: libpressio's noop rejects a typed output buffer, copy directly. */
    if (strcmp(ctx->compressor_id, "noop") == 0) {
        size_t n = csize < out_bytes ? csize : out_bytes;
        memcpy(out, cbuf, n);
        return 0;
    }

    if (vol_is_byte_stream(ctx->compressor_id)) {
        out_dtype    = pressio_byte_dtype;
        out_ndims    = 1;
        byte_dims[0] = out_bytes;
        out_dims     = byte_dims;
    } else {
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            return -1;
        }
        size_t logical = vol_logical_nbytes(ctx);
        if (logical != out_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "shape/size mismatch for '%s': recorded dims imply %zu bytes "
                    "but the read buffer is %zu bytes",
                    ctx->compressor_id, logical, out_bytes);
            return -1;
        }
        out_dtype = ctx->dtype;
        out_ndims = ctx->ndims;
        out_dims  = ctx->dims;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS NATIVE: id=%s csize=%zu out_bytes=%zu ndims=%zu\n",
           ctx->compressor_id, csize, out_bytes, out_ndims);
#endif

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");

    /* Owning output so GPU codecs can place it device-side; we pull it home
     * after the call (same reasoning as transfer_decompress). */
    output = pressio_data_new_owning(out_dtype, out_ndims, out_dims);
    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating %zu-byte decompress output for '%s'",
                out_bytes, ctx->compressor_id);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    vol_set_cuda_stream(ctx);
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
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
            ctx->compress_ms += (double)_ms;
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_derr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "native decompress failed for '%s' (%zu B): %s",
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
                "native decompress failed for '%s' (%zu B): %s",
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
                    "native decompress of '%s' produced no host output",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        if (actual_bytes != out_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "native decompress of '%s' produced %zu bytes, expected %zu",
                    ctx->compressor_id, actual_bytes, out_bytes);
            ret_val = -1;
            goto done;
        }

        memcpy(out, out_ptr, out_bytes);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("DECOMPRESS NATIVE OK: id=%s actual_bytes=%zu\n",
               ctx->compressor_id, actual_bytes);
#endif
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results =
            pressio_compressor_get_metrics_results(ctx->compressor);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] decompress native '%s':\n%s\n", ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

/* ========================================================================
 * VOL-LEVEL CHUNKED PATH (opt-in via VOL_COMP_CHUNK_MB): the write loop
 * slices the buffer and calls these once per chunk. Each chunk is an
 * independent compressed blob recorded in the v2 container header.
 * ======================================================================== */

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
    vol_set_cuda_stream(ctx);
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
    vol_set_cuda_stream(ctx);
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

/* ========================================================================
 * LEGACY WHOLE-BUFFER HELPERS (transfer_compress / transfer_decompress).
 * Kept for callers outside the dataset read/write path. transfer_compress
 * is now just compress_native plus the ctx->compressed_buf bookkeeping.
 * ======================================================================== */

herr_t
H5VL_pass_through_ext_transfer_compress(compression_ctx *ctx, const void *data, size_t nbytes)
{
    void    *cb  = NULL;
    uint64_t len = 0;

    ctx->compress_ms = 0.0;
    if (H5VL_pass_through_ext_compress_native(ctx, data, nbytes, &cb, &len) < 0)
        return -1;

    ctx->compressed_buf        = cb;
    ctx->compressed_chunk_size = (size_t)len;
    return 0;
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
    return H5VL_pass_through_ext_decompress_native(ctx, compressed_data,
                                                   compressed_size, output_buf,
                                                   vol_logical_nbytes(ctx));
}

int
H5VL_pass_through_ext_chunking_mode(const compression_ctx *ctx)
{
    const char *env = getenv("VOL_COMP_CHUNKING");
    if (env && *env) {
        if (strcasecmp(env, "none") == 0 || strcmp(env, "0") == 0)
            return VOL_CHUNKING_NONE;
        if (strcasecmp(env, "vol") == 0)
            return VOL_CHUNKING_VOL;
        if (strcasecmp(env, "pressio") == 0 || strcasecmp(env, "libpressio") == 0)
            return VOL_CHUNKING_PRESSIO;
        /* unrecognized value: ignore the override, fall through */
    }

    if (ctx && ctx->chunking_mode != VOL_CHUNKING_NONE)
        return ctx->chunking_mode;

    /* legacy knob: VOL_COMP_CHUNK_MB alone selects VOL-level chunking */
    env = getenv("VOL_COMP_CHUNK_MB");
    if (env && *env && strtoull(env, NULL, 10) > 0)
        return VOL_CHUNKING_VOL;

    return VOL_CHUNKING_NONE;
}

size_t
H5VL_pass_through_ext_chunk_bytes(const compression_ctx *ctx, size_t dsize)
{
    uint64_t mb = 0;

    const char *env = getenv("VOL_COMP_CHUNK_MB");
    if (env && *env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = (uint64_t)v;
    }
    if (mb == 0 && ctx) mb = ctx->chunk_mb;
    if (mb == 0) mb = 1024;

    if (dsize == 0) dsize = 1;
    size_t bytes = (size_t)mb << 20;
    bytes -= bytes % dsize;             /* whole elements only */
    if (bytes < dsize) bytes = dsize;
    return bytes;
}

void
H5VL_pass_through_ext_parse_chunking_opts(compression_ctx *ctx,
                                          struct pressio_options *opts)
{
    if (!ctx) return;
    ctx->chunking_mode = VOL_CHUNKING_NONE;
    ctx->chunk_mb      = 0;
    if (!opts) return;

    const char *mode = NULL;
    if (pressio_options_get_string(opts, "vol:chunking_mode", &mode) ==
            pressio_options_key_set && mode) {
        if (strcasecmp(mode, "vol") == 0)
            ctx->chunking_mode = VOL_CHUNKING_VOL;
        else if (strcasecmp(mode, "pressio") == 0 ||
                 strcasecmp(mode, "libpressio") == 0)
            ctx->chunking_mode = VOL_CHUNKING_PRESSIO;
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        else if (strcasecmp(mode, "none") != 0)
            fprintf(stderr, "VOL: ignoring unknown vol:chunking_mode '%s'\n", mode);
#endif
        free((void *)mode);
    }

    /* JSON numbers can land as any integer flavor; try them in turn. */
    {
        uint64_t u64 = 0; int64_t i64 = 0; unsigned u32 = 0; int i32 = 0;
        if (pressio_options_get_uinteger64(opts, "vol:chunk_mb", &u64) ==
                pressio_options_key_set)
            ctx->chunk_mb = u64;
        else if (pressio_options_get_integer64(opts, "vol:chunk_mb", &i64) ==
                     pressio_options_key_set && i64 > 0)
            ctx->chunk_mb = (uint64_t)i64;
        else if (pressio_options_get_uinteger(opts, "vol:chunk_mb", &u32) ==
                     pressio_options_key_set)
            ctx->chunk_mb = u32;
        else if (pressio_options_get_integer(opts, "vol:chunk_mb", &i32) ==
                     pressio_options_key_set && i32 > 0)
            ctx->chunk_mb = (uint64_t)i32;
    }
}

herr_t
H5VL_pass_through_ext_compress_pressio(compression_ctx *ctx,
                                       const void *data, size_t nbytes,
                                       size_t chunk_bytes_req,
                                       void **out_cbuf, uint64_t *out_csize,
                                       uint64_t *out_chunk_elems)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    struct pressio_compressor *w = NULL;
    enum pressio_dtype in_dtype;
    size_t elem_size, total_elems, nch;
    uint64_t chunk_elems;
    size_t in_dims[1];
    size_t out_dims[1];

    if (!ctx || !ctx->compressor || !out_cbuf || !out_csize ||
        !out_chunk_elems || !data || nbytes == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "invalid arguments to compress_pressio");
        return -1;
    }
    *out_cbuf        = NULL;
    *out_csize       = 0;
    *out_chunk_elems = 0;

    /* noop's typed-buffer quirk breaks inside the wrapper's per-chunk views;
     * fail loudly instead of producing an unreadable container. */
    if (strcmp(ctx->compressor_id, "noop") == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "chunking_mode 'pressio' does not support the noop codec; "
                "use chunking_mode 'none' or 'vol'");
        return -1;
    }

    /* Same 1-D view convention as the VOL chunked path: shape-aware codecs
     * get typed elements, byte-stream codecs get raw bytes. */
    if (vol_is_byte_stream(ctx->compressor_id)) {
        in_dtype    = pressio_byte_dtype;
        elem_size   = 1;
        total_elems = nbytes;
    } else {
        elem_size = pressio_dtype_size(ctx->dtype);
        if (elem_size == 0 || nbytes % elem_size != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "buffer of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", nbytes, elem_size, ctx->compressor_id);
            return -1;
        }
        in_dtype    = ctx->dtype;
        total_elems = nbytes / elem_size;
    }

    chunk_elems = (uint64_t)(chunk_bytes_req / elem_size);
    if (chunk_elems == 0 || chunk_elems > (uint64_t)total_elems)
        chunk_elems = (uint64_t)total_elems;        /* single chunk */
    nch = (total_elems + (size_t)chunk_elems - 1) / (size_t)chunk_elems;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- COMPRESS PRESSIO: id=%s nbytes=%zu total_elems=%zu "
           "chunk_elems=%llu nchunks=%zu\n",
           ctx->compressor_id, nbytes, total_elems,
           (unsigned long long)chunk_elems, nch);
#endif

    w = vol_get_chunk_wrapper(ctx, chunk_elems);
    if (!w)
        return -1;      /* error already pushed */

    in_dims[0] = total_elems;
    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              1, in_dims,
                                              vol_ptr_domain(data));

    /* Headroom for per-chunk overhead plus the wrapper's own framing. */
    out_dims[0] = nbytes + 4096 + nch * 64;
    output = pressio_data_new_owning(pressio_byte_dtype, 1, out_dims);

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte dataset", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _cerr = pressio_compressor_compress(w, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms += (double)_ms;
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_cerr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "pressio-chunked compress failed for '%s' (%zu B, "
                    "chunk_elems=%llu): %s",
                    ctx->compressor_id, nbytes,
                    (unsigned long long)chunk_elems,
                    pressio_compressor_error_msg(w));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_compress(w, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio-chunked compress failed for '%s' (%zu B, "
                "chunk_elems=%llu): %s",
                ctx->compressor_id, nbytes,
                (unsigned long long)chunk_elems,
                pressio_compressor_error_msg(w));
        ret_val = -1;
        goto done;
    }
#endif

    vol_make_host_resident(output);

    {
        size_t comp_size = 0;
        void  *comp_ptr  = pressio_data_ptr(output, &comp_size);

        if (comp_size == 0 || comp_ptr == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "pressio-chunked compress of '%s' produced 0 bytes for %zu B",
                    ctx->compressor_id, nbytes);
            ret_val = -1;
            goto done;
        }

        void *cb = malloc(comp_size);
        if (!cb) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory copying %zu compressed bytes", comp_size);
            ret_val = -1;
            goto done;
        }

        memcpy(cb, comp_ptr, comp_size);
        *out_cbuf        = cb;
        *out_csize       = (uint64_t)comp_size;
        *out_chunk_elems = chunk_elems;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("COMPRESS PRESSIO OK: id=%s comp_size=%zu original_nbytes=%zu "
               "chunk_elems=%llu\n",
               ctx->compressor_id, comp_size, nbytes,
               (unsigned long long)chunk_elems);
#endif
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results = pressio_compressor_get_metrics_results(w);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] compress pressio-chunked '%s':\n%s\n",
               ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

herr_t
H5VL_pass_through_ext_decompress_pressio(compression_ctx *ctx,
                                         const void *cbuf, size_t csize,
                                         uint64_t chunk_elems,
                                         void *out, size_t out_bytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    struct pressio_compressor *w = NULL;
    enum pressio_dtype out_dtype;
    size_t elem_size, total_elems;
    size_t comp_dims[1];
    size_t out_dims[1];

    if (!ctx || !ctx->compressor || !cbuf || !out || csize == 0 || out_bytes == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid arguments to decompress_pressio");
        return -1;
    }
    if (chunk_elems == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "corrupt pressio-chunked container: chunk_elems=0 in header");
        return -1;
    }
    if (strcmp(ctx->compressor_id, "noop") == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "chunking_mode 'pressio' does not support the noop codec");
        return -1;
    }

    if (vol_is_byte_stream(ctx->compressor_id)) {
        out_dtype   = pressio_byte_dtype;
        elem_size   = 1;
        total_elems = out_bytes;
    } else {
        elem_size = pressio_dtype_size(ctx->dtype);
        if (elem_size == 0 || out_bytes % elem_size != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "output of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", out_bytes, elem_size, ctx->compressor_id);
            return -1;
        }
        out_dtype   = ctx->dtype;
        total_elems = out_bytes / elem_size;
    }

    if (chunk_elems > (uint64_t)total_elems)
        chunk_elems = (uint64_t)total_elems;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS PRESSIO: id=%s csize=%zu out_bytes=%zu "
           "chunk_elems=%llu\n",
           ctx->compressor_id, csize, out_bytes,
           (unsigned long long)chunk_elems);
#endif

    w = vol_get_chunk_wrapper(ctx, chunk_elems);
    if (!w)
        return -1;      /* error already pushed */

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");

    out_dims[0] = total_elems;
    output = pressio_data_new_owning(out_dtype, 1, out_dims);
    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating %zu-byte decompress output for '%s'",
                out_bytes, ctx->compressor_id);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        int _derr = pressio_compressor_decompress(w, input, output);

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->compress_ms += (double)_ms;
            cudaEventDestroy(_ev0);
            cudaEventDestroy(_ev1);
        }

        if (_derr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio-chunked decompress failed for '%s' (%zu B): %s "
                    "(container may be corrupt or written with different options)",
                    ctx->compressor_id, csize,
                    pressio_compressor_error_msg(w));
            ret_val = -1;
            goto done;
        }
    }
#else
    if (pressio_compressor_decompress(w, input, output)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio-chunked decompress failed for '%s' (%zu B): %s "
                "(container may be corrupt or written with different options)",
                ctx->compressor_id, csize,
                pressio_compressor_error_msg(w));
        ret_val = -1;
        goto done;
    }
#endif

    vol_make_host_resident(output);

    {
        size_t actual_bytes = 0;
        void  *out_ptr = pressio_data_ptr(output, &actual_bytes);

        if (!out_ptr || actual_bytes == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio-chunked decompress of '%s' produced no host output",
                    ctx->compressor_id);
            ret_val = -1;
            goto done;
        }
        if (actual_bytes != out_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio-chunked decompress of '%s' produced %zu bytes, "
                    "expected %zu (container corrupt?)",
                    ctx->compressor_id, actual_bytes, out_bytes);
            ret_val = -1;
            goto done;
        }

        memcpy(out, out_ptr, out_bytes);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("DECOMPRESS PRESSIO OK: id=%s actual_bytes=%zu\n",
               ctx->compressor_id, actual_bytes);
#endif
    }

    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *results = pressio_compressor_get_metrics_results(w);
        char *str = pressio_options_to_string(results);
        printf("[VOL METRICS] decompress pressio-chunked '%s':\n%s\n",
               ctx->compressor_id, str);
        free(str);
        pressio_options_free(results);
    }

done:
    if (input)  pressio_data_free(input);
    if (output) pressio_data_free(output);
    return ret_val;
}

} /* extern C */