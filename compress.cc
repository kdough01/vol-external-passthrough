#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <exception>
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

#include <memory>
#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/options.h>
#include <libpressio_ext/cpp/pressio.h>

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

/* ------------------------------------------------------------------------
 * Domain helpers.
 * --------------------------------------------------------------------- */
static int
vol_domain_is_host_accessible(const char *dom)
{
    if (!dom) return 0;
    return strcmp(dom, "malloc")            == 0 ||
           strcmp(dom, "cudamallochost")    == 0 ||
           strcmp(dom, "cudahostalloc")     == 0 ||
           strcmp(dom, "cudamallocmanaged") == 0;
}

static void
vol_make_host_resident(struct pressio_data *data)
{
    if (!data) return;

    if (!vol_domain_is_host_accessible(pressio_data_domain_id(data))) {
        pressio_data *d = data;
        *d = domain_manager().make_readable(
            libpressio::domain_plugins().build("malloc"), std::move(*d));
    }
}

static size_t
vol_full_dims(const compression_ctx *ctx, size_t *out_dims)
{
    for (size_t i = 0; i < ctx->ndims; i++)
        out_dims[i] = ctx->dims[ctx->ndims - 1 - i];
    return ctx->ndims;
}

#ifdef USE_CUDA

static void
vol_make_device_resident(struct pressio_data *data)
{
    if (!data) return;
    if (strcmp(pressio_data_domain_id(data), "cudamalloc") != 0) {
        pressio_data *d = data;
        *d = domain_manager().make_writeable(
            libpressio::domain_plugins().build("cudamalloc"), std::move(*d));
    }
}
#endif

/* Allocate a codec output buffer in the domain where the codec will actually
 * write it.
 */
static struct pressio_data *
vol_new_output(enum pressio_dtype dt, size_t ndims, size_t *dims, int is_gpu)
{
#ifdef USE_CUDA
    if (is_gpu) {
        struct pressio_data *d = pressio_data_new_empty(dt, ndims, dims);
        if (d) vol_make_device_resident(d);
        return d;
    }
#else
    (void)is_gpu;
#endif
    return pressio_data_new_owning(dt, ndims, dims);
}

static int
vol_gpu_events_enabled(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VOL_COMP_GPU_EVENTS");
        v = (e && *e == '0') ? 0 : 1;
    }
    return v;
}

#ifdef USE_CUDA
/* Events are created once per ctx and cached. */
static int
vol_ev_begin(compression_ctx *ctx, int gpu)
{
    if (!gpu || !vol_gpu_events_enabled()) return 0;

    if (!ctx->ev_start || !ctx->ev_stop) {
        cudaEvent_t a = NULL, b = NULL;
        if (cudaEventCreate(&a) != cudaSuccess) return 0;
        if (cudaEventCreate(&b) != cudaSuccess) { cudaEventDestroy(a); return 0; }
        ctx->ev_start = (void *)a;
        ctx->ev_stop  = (void *)b;
    }
    cudaEventRecord((cudaEvent_t)ctx->ev_start, (cudaStream_t)ctx->stream);
    return 1;
}

static void
vol_ev_end(compression_ctx *ctx, int active)
{
    float ms = 0.0f;
    if (!active) return;
    cudaEventRecord((cudaEvent_t)ctx->ev_stop, (cudaStream_t)ctx->stream);
    cudaEventSynchronize((cudaEvent_t)ctx->ev_stop);
    cudaEventElapsedTime(&ms, (cudaEvent_t)ctx->ev_start, (cudaEvent_t)ctx->ev_stop);
    ctx->device_ms += (double)ms;
}

/* Hand the codec our CUDA stream. */
static void
vol_set_cuda_stream(compression_ctx *ctx)
{
    char skey[128];
    int  serr;

    if (ctx->cuda_stream_set) return;

    snprintf(skey, sizeof(skey), "%s:cuda_stream", ctx->compressor_id);

    {
        struct pressio_options *sopt = pressio_options_new();
        pressio_options_set_userptr(sopt, skey, ctx->stream);
        serr = pressio_compressor_set_options(ctx->compressor, sopt);
        pressio_options_free(sopt);
    }

    if (serr) {
        fprintf(stderr, "[VOL stream] set_options('%s') returned %d: %s\n",
                skey, serr, pressio_compressor_error_msg(ctx->compressor));
    }

    {
        struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
        void *rb = NULL;
        enum pressio_options_key_status st = pressio_options_get_userptr(o, skey, &rb);

        if (st != pressio_options_key_set || rb != ctx->stream) {
            ctx->cuda_stream_ok = 0;
            fprintf(stderr,
                "[VOL stream] *** KEY NOT CONFIRMED *** '%s' status=%d "
                "readback=%p expected=%p\n"
                "[VOL stream] GPU event timings (device_ms) are NOT trustworthy, "
                "and D2H ordering is not guaranteed.\n"
                "[VOL stream] Cross-check with: nsys stats --report cuda_gpu_trace\n",
                skey, (int)st, rb, (void *)ctx->stream);

            char *dump = pressio_options_to_string(o);
            fprintf(stderr, "[VOL stream] options declared by '%s':\n%s\n",
                    ctx->compressor_id, dump ? dump : "(null)");
            free(dump);
        } else {
            ctx->cuda_stream_ok = 1;
            if (getenv("VOL_COMP_COPY_LOG"))
                fprintf(stderr, "[VOL stream] confirmed '%s' = %p\n",
                        skey, (void *)ctx->stream);
        }
        pressio_options_free(o);
    }

    ctx->cuda_stream_set = 1;
}
#endif /* USE_CUDA */

#ifdef USE_CUDA

static cudaError_t
vol_stream_barrier(compression_ctx *ctx)
{
    cudaError_t cerr = cudaGetLastError();      /* surface prior async faults */
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[VOL] pending CUDA error before D2H: %s\n",
                cudaGetErrorString(cerr));
        return cerr;
    }

    if (!ctx->cuda_stream_ok && !ctx->stream_warn_done) {
        fprintf(stderr,
            "[VOL] *** D2H ORDERING NOT GUARANTEED *** the '%s:cuda_stream' "
            "option was not confirmed, so the codec may be running on a stream "
            "we do not synchronise. Results may be read before the kernel "
            "finishes. Fix the option key; do not mask this with a device "
            "barrier.\n", ctx->compressor_id);
        ctx->stream_warn_done = 1;
    }

    return cudaStreamSynchronize((cudaStream_t)ctx->stream);
}
#endif

/* One explicit, correctly sized copy of the compressed result into a host
 * buffer */
static int
vol_fetch_result(compression_ctx *ctx, struct pressio_data *result,
                 size_t hdr_reserve, void **out_base, size_t *out_size)
{
    size_t sz  = 0;
    void  *src = pressio_data_ptr(result, &sz);
    void  *dst = NULL;

    *out_base = NULL;
    *out_size = 0;

    if (!src || sz == 0) return -1;

    dst = malloc(hdr_reserve + sz);
    if (!dst) return -2;

#ifdef USE_CUDA
    if (H5VL_pass_through_ext_buf_is_device(src)) {
        cudaError_t cerr = cudaMemcpyAsync((char *)dst + hdr_reserve, src, sz,
                                           cudaMemcpyDeviceToHost,
                                           (cudaStream_t)ctx->stream);
        if (cerr == cudaSuccess)
            cerr = vol_stream_barrier(ctx);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[VOL] D2H of %zu compressed bytes failed: %s\n",
                    sz, cudaGetErrorString(cerr));
            free(dst);
            return -3;
        }
    } else {
        memcpy((char *)dst + hdr_reserve, src, sz);
    }
#else
    (void)ctx;
    memcpy((char *)dst + hdr_reserve, src, sz);
#endif

    *out_base = dst;
    *out_size = sz;
    return 0;
}

/* Copy a decompressed result into the caller's buffer with a single transfer,
 * instead of letting libpressio stage it through a host buffer and then
 * memcpy'ing the full logical size again. */
static int
vol_fetch_into(compression_ctx *ctx, struct pressio_data *result,
               void *dst, size_t want)
{
    size_t sz  = 0;
    void  *src = pressio_data_ptr(result, &sz);

    if (!src || sz == 0) return -1;
    if (sz != want)      return -2;

#ifdef USE_CUDA
    if (H5VL_pass_through_ext_buf_is_device(src)) {
        cudaError_t cerr = cudaMemcpyAsync(dst, src, want, cudaMemcpyDeviceToHost,
                                           (cudaStream_t)ctx->stream);
        if (cerr == cudaSuccess)
            cerr = vol_stream_barrier(ctx);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[VOL] D2H of %zu decompressed bytes failed: %s\n",
                    want, cudaGetErrorString(cerr));
            return -3;
        }
        return 0;
    }
#else
    (void)ctx;
#endif
    memcpy(dst, src, want);
    return 0;
}

static size_t
vol_slab_dims(const compression_ctx *ctx, size_t nelem, hid_t minor,
              size_t *out_dims)
{
    size_t plane, nslices, k;
 
    if (!ctx || ctx->ndims == 0 || ctx->dims == NULL) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "compressor '%s' is shape-aware but no dtype/shape was "
                "recorded for this dataset",
                ctx ? ctx->compressor_id : "?");
        return 0;
    }
    if (nelem == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "empty slab requested for '%s'", ctx->compressor_id);
        return 0;
    }
 
    plane = 1;                          /* elements in one slowest-axis slice */
    for (size_t i = 1; i < ctx->ndims; i++)
        plane *= ctx->dims[i];
 
    if (plane == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "degenerate shape recorded for '%s' (a fast dimension is 0)",
                ctx->compressor_id);
        return 0;
    }
    if (nelem % plane != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "slab of %zu elements is not a whole number of %zu-element "
                "slices for '%s'; chunk sizes must be a multiple of one slice",
                nelem, plane, ctx->compressor_id);
        return 0;
    }
    nslices = nelem / plane;
 
    k = 0;
    for (size_t i = ctx->ndims; i-- > 1; )
        out_dims[k++] = ctx->dims[i];
    out_dims[k++] = nslices;
 
    return k;
}
 
/* Byte-count form, used by the VOL-chunked path. */
static size_t
vol_chunk_dims(const compression_ctx *ctx, size_t nbytes, hid_t minor,
               size_t *out_dims)
{
    size_t dsize;

    if (!ctx) return 0;
    dsize = pressio_dtype_size(ctx->dtype);
    if (dsize == 0 || nbytes == 0 || nbytes % dsize != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "chunk of %zu bytes is not a whole number of %zu-byte "
                "elements for '%s'", nbytes, dsize, ctx->compressor_id);
        return 0;
    }
    return vol_slab_dims(ctx, nbytes / dsize, minor, out_dims);
}

static struct pressio_compressor *
vol_get_chunk_wrapper(compression_ctx *ctx, uint64_t chunk_elems)
{
    size_t cdims[H5S_MAX_RANK];
    size_t cnd;
 
    if (ctx->chunk_wrapper && ctx->chunk_wrapper_elems == chunk_elems)
        return ctx->chunk_wrapper;
 
    if (ctx->chunk_wrapper) {
        pressio_compressor_release(ctx->chunk_wrapper);
        ctx->chunk_wrapper       = NULL;
        ctx->chunk_wrapper_elems = 0;
    }
 
    /* Chunk shape, in libpressio order. Byte-stream codecs stay flat. */
    if (vol_is_byte_stream(ctx->compressor_id)) {
        cdims[0] = (size_t)chunk_elems;
        cnd      = 1;
    } else {
        cnd = vol_slab_dims(ctx, (size_t)chunk_elems, min_compress_failed, cdims);
        if (cnd == 0)
            return NULL;                /* error already pushed */
    }
 
    struct pressio *lib = pressio_instance();
    struct pressio_compressor *w = lib ? pressio_get_compressor(lib, "chunking") : NULL;
    if (lib) pressio_release(lib);
    if (!w) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compressor_unavail,
                "libpressio 'chunking' meta-compressor is not available in "
                "this build; use chunking_mode 'none' or 'vol' instead");
        return NULL;
    }
 
    {
        struct pressio_options *nest = pressio_options_new();
        pressio_options_set_string(nest, "chunking:compressor",
                                   "many_independent_threaded");
        pressio_options_set_string(nest, "many_independent_threaded:compressor",
                                   "vol_host_output");
        pressio_options_set_string(nest, "vol_host_output:compressor",
                                   ctx->compressor_id);
        (void)pressio_compressor_set_options(w, nest);
        pressio_options_free(nest);
 
        struct pressio_options *chk = pressio_compressor_get_options(w);
        char *cs = pressio_options_to_string(chk);
        int ok = (cs && strstr(cs, "many_independent_threaded") != NULL
                     && strstr(cs, "vol_host_output") != NULL);
        free(cs);
        pressio_options_free(chk);
        if (!ok) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "chunking pipeline failed to assemble "
                    "many_independent_threaded/vol_host_output for '%s' -- "
                    "refusing to store uncompressed data", ctx->compressor_id);
            pressio_compressor_release(w);
            return NULL;
        }
    }

    {
        struct pressio_options *opts =
            pressio_compressor_get_options(ctx->compressor);
 
        size_t csz_len = cnd;
        size_t csz_bytes = 0;
        struct pressio_data *csz =
            pressio_data_new_owning(pressio_uint64_dtype, 1, &csz_len);
        if (!csz) {
            pressio_options_free(opts);
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory building chunking:size for '%s'",
                    ctx->compressor_id);
            pressio_compressor_release(w);
            return NULL;
        }
        {
            uint64_t *p = (uint64_t *)pressio_data_ptr(csz, &csz_bytes);
            for (size_t i = 0; i < cnd; i++)
                p[i] = (uint64_t)cdims[i];
        }
        pressio_options_set_data(opts, "chunking:size", csz);
        pressio_data_free(csz);
 
        const char *nt = getenv("VOL_COMP_PRESSIO_NTHREADS");
        if (nt && *nt) {
            unsigned long v = strtoul(nt, NULL, 10);
            if (v > 0)
                pressio_options_set_uinteger(opts,
                    "many_independent_threaded:nthreads", (unsigned)v);
        }
 
        int serr = pressio_compressor_set_options(w, opts);
        pressio_options_free(opts);
        if (serr) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compressor_unavail,
                    "failed to set chunking:size (rank %zu) on wrapper: %s",
                    cnd, pressio_compressor_error_msg(w));
            pressio_compressor_release(w);
            return NULL;
        }
    }
 
#ifdef USE_CUDA
    if (ctx->stream) {
        struct pressio_options *s = pressio_options_new();
        char skey[128];
        snprintf(skey, sizeof(skey), "%s:cuda_stream", ctx->compressor_id);
        pressio_options_set_userptr(s, skey, ctx->stream);
        (void)pressio_compressor_set_options(w, s);
        pressio_options_free(s);
    }
#endif
 
    if (getenv("VOL_COMP_CHUNK_LOG")) {
        fprintf(stderr, "[chunk] pressio-mode chunking:size = [");
        for (size_t i = 0; i < cnd; i++)
            fprintf(stderr, "%zu%s", cdims[i], (i + 1 < cnd) ? "," : "");
        fprintf(stderr, "] (pressio order, fastest first)\n");
    }
 
    if (getenv("HDF5_VOL_DUMP_PIPELINE")) {
        struct pressio_options *built = pressio_compressor_get_options(w);
        char *s2 = pressio_options_to_string(built);
        fprintf(stderr, "[VOL BUILT PIPELINE '%s']\n%s\n",
                ctx->compressor_id, s2 ? s2 : "(null)");
        free(s2);
        pressio_options_free(built);
    }
 
    ctx->chunk_wrapper       = w;
    ctx->chunk_wrapper_elems = chunk_elems;
    return w;
}

static const char *
vol_ptr_domain(const void *p)
{
    return H5VL_pass_through_ext_buf_is_device(p) ? "cudamalloc" : "malloc";
}

/* Defined inside libpressio's own namespaces so the plugin API names
 * resolve exactly as they do in libpressio's in-tree plugins. */
namespace libpressio { namespace compressors { namespace vol_host_output_ns {

struct vol_host_output_plugin final : public libpressio_compressor_plugin {
    pressio_compressor child    = compressor_plugins().build("noop");
    std::string        child_id = "noop";

    struct pressio_options get_options_impl() const override {
        pressio_options opts;
        set_meta(opts, "vol_host_output:compressor", child_id, child);
        return opts;
    }
    int set_options_impl(pressio_options const& opts) override {
        get_meta(opts, "vol_host_output:compressor", compressor_plugins(),
                 child_id, child);
        return 0;
    }
    struct pressio_options get_configuration_impl() const override {
        pressio_options opts;
        set_meta_configuration(opts, "vol_host_output:compressor",
                               compressor_plugins(), child);
        set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
        set(opts, "pressio:stability", "external");
        return opts;
    }
    struct pressio_options get_documentation_impl() const override {
        pressio_options opts;
        set_meta_docs(opts, "vol_host_output:compressor",
                      "child codec whose outputs are forced host-resident", child);
        set(opts, "pressio:description",
            "forces the child compressor's output into the malloc domain");
        return opts;
    }
    pressio_options get_metrics_results_impl() const override {
        return child->get_metrics_results();
    }
    int compress_impl(const pressio_data* input, pressio_data* output) override {
        int rc = child->compress(input, output);
        if (rc) return set_error(child->error_code(), child->error_msg());
        vol_make_host_resident(output);      /* no-op if already host-accessible */
        return 0;
    }
    int decompress_impl(const pressio_data* input, pressio_data* output) override {
        int rc = child->decompress(input, output);
        if (rc) return set_error(child->error_code(), child->error_msg());
        vol_make_host_resident(output);
        return 0;
    }
    void set_name_impl(std::string const& new_name) override {
        if (!new_name.empty()) child->set_name(new_name + "/" + child->prefix());
        else                   child->set_name(new_name);
    }
    std::vector<std::string> children_impl() const final {
        return { child->get_name() };
    }
    const char* prefix() const override  { return "vol_host_output"; }
    const char* version() const override { return "0.0.1"; }
    int major_version() const override { return 0; }
    int minor_version() const override { return 0; }
    int patch_version() const override { return 1; }
    std::shared_ptr<libpressio_compressor_plugin> clone() override {
        return std::make_shared<vol_host_output_plugin>(*this);
    }
};

pressio_register vol_host_output_registration(
    compressor_plugins(), "vol_host_output",
    [] { return std::make_shared<vol_host_output_plugin>(); });

} } } /* namespace */

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

/* ========================================================================
 * NATIVE PATH (default): one compress/decompress call over the whole
 * dataset, straight into ctx->compressor. No chunking wrapper of any kind;
 * whatever chunking happens is the codec's own internal blocking.
 * ======================================================================== */

static herr_t
vol_compress_native_impl(compression_ctx *ctx,
                         const void *data, size_t nbytes,
                         size_t hdr_reserve,
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
    size_t  rdims[H5S_MAX_RANK];
    size_t  logical;
    int     is_gpu = 0;
    int     ev     = 0;
    int     cerr   = 0;
    int     frc    = 0;
    void   *base   = NULL;
    size_t  csize  = 0;
    double  t0     = 0.0;

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
        logical = vol_logical_nbytes(ctx);
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
        for (size_t i = 0; i < ctx->ndims; i++)
            rdims[i] = ctx->dims[ctx->ndims - 1 - i];
        in_dims = rdims;
    }

    is_gpu = vol_is_gpu_codec(ctx->compressor_id);

    if (getenv("VOL_COMP_DUMP_CALL")) {                 /* <-- insert */
        struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
        char *s = pressio_options_to_string(o);
        fprintf(stderr, "[VOL CALL] native id=%s dtype=%d rank=%zu dims=[",
                ctx->compressor_id, (int)in_dtype, in_ndims);
        for (size_t i = 0; i < in_ndims; i++)
            fprintf(stderr, "%zu%s", in_dims[i], (i + 1 < in_ndims) ? "," : "");
        fprintf(stderr, "] nbytes=%zu\n[VOL CALL] options:\n%s\n",
                nbytes, s ? s : "(null)");
        free(s);
        pressio_options_free(o);
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

    out_dims[0] = nbytes + nbytes / 8 + (1u << 16);
    output = vol_new_output(pressio_byte_dtype, 1, out_dims, is_gpu);

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte dataset", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    vol_set_cuda_stream(ctx);
#endif

#ifdef USE_CUDA
    ev = vol_ev_begin(ctx, is_gpu);
#endif
    t0   = bench_now_ms();
    cerr = pressio_compressor_compress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;
#ifdef USE_CUDA
    vol_ev_end(ctx, ev);
#endif

    if (cerr) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compress failed for '%s' (%zu B): %s",
                ctx->compressor_id, nbytes,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }

    frc = vol_fetch_result(ctx, output, hdr_reserve, &base, &csize);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compress of '%s' produced no usable output for %zu B "
                "(fetch rc=%d)", ctx->compressor_id, nbytes, frc);
        ret_val = -1;
        goto done;
    }

    if (csize >= (size_t)out_dims[0]) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' filled the entire %zu byte output buffer for "
                "%zu bytes of input -- the result is truncated and would decode "
                "to garbage", ctx->compressor_id, (size_t)out_dims[0], nbytes);
        free(base);
        ret_val = -1;
        goto done;
    }
    if (csize >= nbytes)
        fprintf(stderr,
            "[VOL WARN] '%s' EXPANDED %zu bytes to %zu bytes (%.4f%% of input). "
            "Headroom was %zu bytes.\n",
            ctx->compressor_id, nbytes, csize,
            100.0 * (double)csize / (double)nbytes, (size_t)out_dims[0] - nbytes);
    else if (getenv("VOL_COMP_COPY_LOG"))
        fprintf(stderr, "[copy] native id=%-10s domain=%-12s payload=%zu B "
                        "declared_cap=%zu B reserve=%zu B\n",
                ctx->compressor_id, pressio_data_domain_id(output),
                csize, (size_t)out_dims[0], hdr_reserve);

    *out_cbuf  = base;
    *out_csize = (uint64_t)csize;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("COMPRESS NATIVE OK: id=%s comp_size=%zu original_nbytes=%zu\n",
           ctx->compressor_id, csize, nbytes);
#endif

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
H5VL_pass_through_ext_compress_native(compression_ctx *ctx,
                                      const void *data, size_t nbytes,
                                      size_t hdr_reserve,
                                      void **out_cbuf, uint64_t *out_csize)
{
    try {
        return vol_compress_native_impl(ctx, data, nbytes, hdr_reserve,
                                        out_cbuf, out_csize);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

static herr_t
vol_decompress_native_impl(compression_ctx *ctx,
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
    size_t  rdims[H5S_MAX_RANK];
    size_t  logical;
    int     is_gpu = 0;
    int     ev     = 0;
    int     derr   = 0;
    int     frc    = 0;
    double  t0     = 0.0;

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
        logical = vol_logical_nbytes(ctx);
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
        for (size_t i = 0; i < ctx->ndims; i++)
            rdims[i] = ctx->dims[ctx->ndims - 1 - i];
        out_dims = rdims;
    }

    is_gpu = vol_is_gpu_codec(ctx->compressor_id);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS NATIVE: id=%s csize=%zu out_bytes=%zu ndims=%zu\n",
           ctx->compressor_id, csize, out_bytes, out_ndims);
#endif

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");

    output = vol_new_output(out_dtype, out_ndims, out_dims, is_gpu);
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
    ev = vol_ev_begin(ctx, is_gpu);
#endif
    t0   = bench_now_ms();
    derr = pressio_compressor_decompress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;
#ifdef USE_CUDA
    vol_ev_end(ctx, ev);
#endif

    if (derr) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "native decompress failed for '%s' (%zu B): %s",
                ctx->compressor_id, csize,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }

    /* One transfer, straight into the caller's buffer. */
    frc = vol_fetch_into(ctx, output, out, out_bytes);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "native decompress of '%s' produced the wrong output "
                "(expected %zu bytes, fetch rc=%d)",
                ctx->compressor_id, out_bytes, frc);
        ret_val = -1;
        goto done;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("DECOMPRESS NATIVE OK: id=%s out_bytes=%zu\n",
           ctx->compressor_id, out_bytes);
#endif

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

herr_t
H5VL_pass_through_ext_decompress_native(compression_ctx *ctx,
                                        const void *cbuf, size_t csize,
                                        void *out, size_t out_bytes)
{
    try {
        return vol_decompress_native_impl(ctx, cbuf, csize, out, out_bytes);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "native decompress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "native decompress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

/* ========================================================================
 * VOL-LEVEL CHUNKED PATH: the write loop slices the buffer and calls these
 * once per chunk. Each chunk is an independent compressed blob recorded in
 * the v2 container header.
 * ======================================================================== */
static herr_t
vol_transfer_compress_chunk_impl(compression_ctx *ctx,
                                 const void *data, size_t nbytes,
                                 void **out_cbuf, uint64_t *out_csize)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype in_dtype;
    size_t chunk_dims[H5S_MAX_RANK];
    size_t in_ndims;
    size_t out_dims[1];
    int    is_gpu = 0;
    int    ev     = 0;
    int    cerr   = 0;
    int    frc    = 0;
    void  *base   = NULL;
    size_t csize  = 0;
    double t0     = 0.0;
 
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
        in_ndims      = 1;
        chunk_dims[0] = nbytes;
    } else {
        in_dtype = ctx->dtype;
        in_ndims = vol_chunk_dims(ctx, nbytes, min_compress_failed, chunk_dims);
        if (in_ndims == 0)
            return -1;              /* error already pushed */
    }
 
    is_gpu = vol_is_gpu_codec(ctx->compressor_id);
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- COMPRESS CHUNK: id=%s nbytes=%zu rank=%zu dims=[",
           ctx->compressor_id, nbytes, in_ndims);
    for (size_t i = 0; i < in_ndims; i++)
        printf("%zu%s", chunk_dims[i], (i + 1 < in_ndims) ? "," : "");
    printf("] (pressio order, fastest first) dtype=%d\n", (int)in_dtype);
#endif
 
    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              in_ndims, chunk_dims,
                                              vol_ptr_domain(data));
 
    out_dims[0] = nbytes + nbytes / 8 + (1u << 16);
    output = vol_new_output(pressio_byte_dtype, 1, out_dims, is_gpu);
 
    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte chunk", nbytes);
        ret_val = -1;
        goto done;
    }
 
#ifdef USE_CUDA
    vol_set_cuda_stream(ctx);
    ev = vol_ev_begin(ctx, is_gpu);
#endif
    t0   = bench_now_ms();
    cerr = pressio_compressor_compress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* ACCUMULATE across chunks */
#ifdef USE_CUDA
    vol_ev_end(ctx, ev);
#endif
 
    if (cerr) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio_compressor_compress failed for '%s' (chunk of %zu B): %s",
                ctx->compressor_id, nbytes,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
 
    frc = vol_fetch_result(ctx, output, 0, &base, &csize);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' produced no usable output for a %zu byte chunk "
                "(fetch rc=%d)", ctx->compressor_id, nbytes, frc);
        ret_val = -1;
        goto done;
    }
 
    /* A chunk that did not compress is the signature of the overflow above.
     * Refuse to store it rather than write a container that decodes to garbage:
     * silent corruption at read time is far worse than a loud failure here. */
    if (csize >= (size_t)out_dims[0]) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' filled the entire %zu byte output buffer for a "
                "%zu byte chunk -- the result is truncated and would decode to "
                "garbage. Increase the headroom in vol_transfer_compress_chunk_impl",
                ctx->compressor_id, (size_t)out_dims[0], nbytes);
        free(base);
        ret_val = -1;
        goto done;
    }
    if (csize >= nbytes)
        fprintf(stderr,
            "[VOL WARN] '%s' EXPANDED a %zu byte chunk to %zu bytes (%.4f%% of "
            "input). Headroom was %zu bytes.\n",
            ctx->compressor_id, nbytes, csize,
            100.0 * (double)csize / (double)nbytes, (size_t)out_dims[0] - nbytes);
    else if (getenv("VOL_COMP_CHUNK_LOG"))
        fprintf(stderr, "[chunk-out] '%s' rank=%zu %zu -> %zu B (%.2fx), cap %zu B\n",
                ctx->compressor_id, in_ndims, nbytes, csize,
                (double)nbytes / (double)csize, (size_t)out_dims[0]);
 
    *out_cbuf  = base;
    *out_csize = (uint64_t)csize;
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("COMPRESS CHUNK OK: id=%s comp_size=%zu original_nbytes=%zu\n",
           ctx->compressor_id, csize, nbytes);
#endif
 
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
H5VL_pass_through_ext_transfer_compress_chunk(compression_ctx *ctx,
                                              const void *data, size_t nbytes,
                                              void **out_cbuf, uint64_t *out_csize)
{
    try {
        return vol_transfer_compress_chunk_impl(ctx, data, nbytes,
                                                out_cbuf, out_csize);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "chunk compress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "chunk compress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}
 
 
/* ========================================================================
 * 3. DECOMPRESS ONE VOL-LEVEL CHUNK
 * ======================================================================== */
static herr_t
vol_transfer_decompress_chunk_impl(compression_ctx *ctx,
                                   const void *cbuf, size_t csize,
                                   void *out, size_t out_bytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    enum pressio_dtype out_dtype;
    size_t comp_dims[1];
    size_t chunk_dims[H5S_MAX_RANK];
    size_t out_ndims;
    int    is_gpu = 0;
    int    ev     = 0;
    int    derr   = 0;
    int    frc    = 0;
    double t0     = 0.0;
 
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
        out_dtype     = pressio_byte_dtype;
        out_ndims     = 1;
        chunk_dims[0] = out_bytes;
    } else {
        /* Must mirror the write side exactly: the codec decodes against the
         * shape it was given at compress time. */
        out_dtype = ctx->dtype;
        out_ndims = vol_chunk_dims(ctx, out_bytes, min_decompress_failed,
                                   chunk_dims);
        if (out_ndims == 0)
            return -1;              /* error already pushed */
    }
 
    is_gpu = vol_is_gpu_codec(ctx->compressor_id);
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS CHUNK: id=%s csize=%zu out_bytes=%zu rank=%zu "
           "dims=[", ctx->compressor_id, csize, out_bytes, out_ndims);
    for (size_t i = 0; i < out_ndims; i++)
        printf("%zu%s", chunk_dims[i], (i + 1 < out_ndims) ? "," : "");
    printf("]\n");
#endif
 
    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");
 
    output = vol_new_output(out_dtype, out_ndims, chunk_dims, is_gpu);
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
    ev = vol_ev_begin(ctx, is_gpu);
#endif
    t0   = bench_now_ms();
    derr = pressio_compressor_decompress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* ACCUMULATE across chunks */
#ifdef USE_CUDA
    vol_ev_end(ctx, ev);
#endif
 
    if (derr) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio_compressor_decompress failed for '%s' (chunk of %zu B): %s",
                ctx->compressor_id, csize,
                pressio_compressor_error_msg(ctx->compressor));
        ret_val = -1;
        goto done;
    }
 
    frc = vol_fetch_into(ctx, output, out, out_bytes);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "chunk decompress of '%s' produced the wrong output "
                "(expected %zu bytes, fetch rc=%d)",
                ctx->compressor_id, out_bytes, frc);
        ret_val = -1;
        goto done;
    }
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("DECOMPRESS CHUNK OK: id=%s out_bytes=%zu\n",
           ctx->compressor_id, out_bytes);
#endif
 
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
    try {
        return vol_transfer_decompress_chunk_impl(ctx, cbuf, csize,
                                                  out, out_bytes);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "chunk decompress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "chunk decompress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

/* ========================================================================
 * LEGACY WHOLE-BUFFER HELPERS (transfer_compress / transfer_decompress).
 * Kept for callers outside the dataset read/write path. transfer_compress
 * is now just compress_native plus the ctx->compressed_buf bookkeeping.
 * These call the already-wrapped entry points, so no try/catch needed.
 * ======================================================================== */

herr_t
H5VL_pass_through_ext_transfer_compress(compression_ctx *ctx, const void *data, size_t nbytes)
{
    void    *cb  = NULL;
    uint64_t len = 0;

    ctx->device_ms       = 0.0;
    ctx->pressio_call_ms = 0.0;
    if (H5VL_pass_through_ext_compress_native(ctx, data, nbytes, 0, &cb, &len) < 0)
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

    env = getenv("VOL_COMP_CHUNK_N");
    if (env && *env && strtoull(env, NULL, 10) > 0)
        return VOL_CHUNKING_VOL;

    return VOL_CHUNKING_NONE;
}

size_t
H5VL_pass_through_ext_chunk_bytes(const compression_ctx *ctx,
                                  size_t total_bytes, size_t elem_size)
{
    uint64_t n = 0;
    size_t   plane = 1;             /* elements per slowest-axis slice */
    size_t   total_elems, chunk_elems, chunk_bytes, nch;
    int      exact;
 
    const char *env = getenv("VOL_COMP_CHUNK_N");
    if (env && *env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) n = (uint64_t)v;
    }
    if (n == 0 && ctx) n = ctx->chunk_n;
    if (n == 0) n = 1;                        /* default: single chunk */
 
    if (elem_size == 0)   elem_size = 1;
    if (total_bytes == 0) total_bytes = elem_size;
 
    total_elems = total_bytes / elem_size;
    if (total_elems == 0) total_elems = 1;
 
    /* Shape-aware codecs get slice granularity; byte-stream codecs are flat. */
    if (ctx && ctx->dims && ctx->ndims > 1 &&
        !vol_is_byte_stream(ctx->compressor_id)) {
        for (size_t i = 1; i < ctx->ndims; i++)
            plane *= ctx->dims[i];
        if (plane == 0 || plane > total_elems || total_elems % plane != 0) {
            fprintf(stderr,
                "[VOL WARN] chunk sizing for '%s': recorded shape implies "
                "%zu-element slices but the buffer holds %zu elements; "
                "falling back to flat chunking. Chunked ratios will not be "
                "comparable to the HDF5 filter.\n",
                ctx->compressor_id, plane, total_elems);
            plane = 1;
        }
    }
 
    {
        size_t total_slices = total_elems / plane;
        size_t slices_per_chunk;
 
        if (n > total_slices) n = total_slices;   /* at most 1 slice/chunk */
        if (n == 0)           n = 1;
 
        slices_per_chunk = (total_slices + n - 1) / n;
        if (slices_per_chunk == 0) slices_per_chunk = 1;
 
        chunk_elems = slices_per_chunk * plane;
        chunk_bytes = chunk_elems * elem_size;
 
        /* actual layout (differs from n only if n doesn't divide evenly) */
        nch   = (total_slices + slices_per_chunk - 1) / slices_per_chunk;
        exact = (total_slices % slices_per_chunk == 0);
    }
 
    if (getenv("VOL_COMP_CHUNK_LOG"))
        fprintf(stderr,
            "[chunk] %-10s N=%-3llu -> %zu chunks x %zu B (%.3f MiB, %zu elems,"
            " %zu slices of %zu elems) total=%zu B%s\n",
            ctx ? ctx->compressor_id : "?", (unsigned long long)n,
            nch, chunk_bytes, chunk_bytes / (1024.0 * 1024.0),
            chunk_elems, chunk_elems / plane, plane, total_bytes,
            exact ? "" : "  [RAGGED: final chunk is short]");
 
    return chunk_bytes;
}

void
H5VL_pass_through_ext_parse_chunking_opts(compression_ctx *ctx,
                                          struct pressio_options *opts)
{
    if (!ctx) return;
    ctx->chunking_mode = VOL_CHUNKING_NONE;
    ctx->chunk_n      = 0;
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
        if (pressio_options_get_uinteger64(opts, "vol:chunk_n", &u64) ==
                pressio_options_key_set)
            ctx->chunk_n = u64;
        else if (pressio_options_get_integer64(opts, "vol:chunk_n", &i64) ==
                     pressio_options_key_set && i64 > 0)
            ctx->chunk_n = (uint64_t)i64;
        else if (pressio_options_get_uinteger(opts, "vol:chunk_n", &u32) ==
                     pressio_options_key_set)
            ctx->chunk_n = u32;
        else if (pressio_options_get_integer(opts, "vol:chunk_n", &i32) ==
                     pressio_options_key_set && i32 > 0)
            ctx->chunk_n = (uint64_t)i32;
    }
}

/* ========================================================================
 * PRESSIO CHUNKED PATH
 * ======================================================================== */

static herr_t
vol_compress_pressio_impl(compression_ctx *ctx,
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
    size_t elem_size, total_elems, plane, nch;
    uint64_t chunk_elems;
    size_t in_dims[H5S_MAX_RANK];
    size_t in_ndims;
 
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
 
    if (vol_is_byte_stream(ctx->compressor_id)) {
        in_dtype    = pressio_byte_dtype;
        elem_size   = 1;
        total_elems = nbytes;
        in_ndims    = 1;
        in_dims[0]  = total_elems;
        plane       = 1;
    } else {
        elem_size = pressio_dtype_size(ctx->dtype);
        if (elem_size == 0 || nbytes % elem_size != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "buffer of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", nbytes, elem_size, ctx->compressor_id);
            return -1;
        }
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            return -1;
        }
        in_dtype    = ctx->dtype;
        total_elems = nbytes / elem_size;
 
        /* The input must carry the FULL shape: chunking compares the chunk
         * rank against the input rank, and slices it N-dimensionally. */
        in_ndims = vol_full_dims(ctx, in_dims);
 
        if (vol_logical_nbytes(ctx) != nbytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "shape/size mismatch for '%s': recorded dims imply %zu bytes "
                    "but the write buffer is %zu bytes",
                    ctx->compressor_id, vol_logical_nbytes(ctx), nbytes);
            return -1;
        }
 
        plane = 1;
        for (size_t i = 1; i < ctx->ndims; i++)
            plane *= ctx->dims[i];
    }
 
    /* Round the requested chunk size DOWN to a whole slab, exactly as the
     * VOL-chunked path does, so both modes split the array identically. */
    {
        size_t req_elems = chunk_bytes_req / elem_size;
        size_t slices_total, slices_req;
 
        if (plane == 0) plane = 1;
        slices_total = total_elems / plane;
        if (slices_total == 0) slices_total = 1;
 
        slices_req = req_elems / plane;
        if (slices_req == 0)            slices_req = 1;
        if (slices_req > slices_total)  slices_req = slices_total;
 
        if (slices_total % slices_req != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "chunking_mode 'pressio': %zu slices along the slowest axis "
                    "is not evenly divisible by %zu slices per chunk; "
                    "libpressio's chunking plugin would pad the final chunk, "
                    "which is not bound-safe for all codecs (observed with "
                    "cuszp). Pick a chunk count that divides %zu, or use "
                    "chunking_mode 'vol'",
                    slices_total, slices_req, slices_total);
            return -1;
        }
 
        chunk_elems = (uint64_t)(slices_req * plane);
        nch         = slices_total / slices_req;
    }
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- COMPRESS PRESSIO: id=%s nbytes=%zu rank=%zu dims=[",
           ctx->compressor_id, nbytes, in_ndims);
    for (size_t i = 0; i < in_ndims; i++)
        printf("%zu%s", in_dims[i], (i + 1 < in_ndims) ? "," : "");
    printf("] chunk_elems=%llu nchunks=%zu\n",
           (unsigned long long)chunk_elems, nch);
#else
    (void)nch;
#endif
 
    w = vol_get_chunk_wrapper(ctx, chunk_elems);
    if (!w)
        return -1;      /* error already pushed */
 
    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              in_ndims, in_dims,
                                              vol_ptr_domain(data));
 
    /* Empty output: the chunking pipeline allocates and sizes the result
     * itself. A preallocated buffer is kept at full capacity (ratio 1.0x)
     * and its layout doesn't round-trip. */
    output = pressio_data_new_empty(pressio_byte_dtype, 0, NULL);
 
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
        double _t0;
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }
 
        _t0 = bench_now_ms();
        int _cerr = pressio_compressor_compress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;
 
        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->device_ms += (double)_ms;
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
    {
        double _t0 = bench_now_ms();
        int _cerr = pressio_compressor_compress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;
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
H5VL_pass_through_ext_compress_pressio(compression_ctx *ctx,
                                       const void *data, size_t nbytes,
                                       size_t chunk_bytes_req,
                                       void **out_cbuf, uint64_t *out_csize,
                                       uint64_t *out_chunk_elems)
{
    try {
        return vol_compress_pressio_impl(ctx, data, nbytes, chunk_bytes_req,
                                         out_cbuf, out_csize, out_chunk_elems);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio-chunked compress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio-chunked compress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

/* ========================================================================
 * 3. DECOMPRESS -- pressio-chunked
 * ======================================================================== */
static herr_t
vol_decompress_pressio_impl(compression_ctx *ctx,
                            const void *cbuf, size_t csize,
                            uint64_t chunk_elems,
                            void *out, size_t out_bytes)
{
    herr_t ret_val = 0;
    struct pressio_data *input  = NULL;
    struct pressio_data *output = NULL;
    struct pressio_compressor *w = NULL;
    enum pressio_dtype out_dtype;
    size_t elem_size, total_elems, plane;
    size_t comp_dims[1];
    size_t out_dims[H5S_MAX_RANK];
    size_t out_ndims;
 
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
        out_ndims   = 1;
        out_dims[0] = total_elems;
        plane       = 1;
    } else {
        elem_size = pressio_dtype_size(ctx->dtype);
        if (elem_size == 0 || out_bytes % elem_size != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "output of %zu bytes is not a whole number of %zu-byte "
                    "elements for '%s'", out_bytes, elem_size, ctx->compressor_id);
            return -1;
        }
        if (ctx->ndims == 0 || ctx->dims == NULL) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "compressor '%s' needs dtype/shape, but none recorded",
                    ctx->compressor_id);
            return -1;
        }
        out_dtype   = ctx->dtype;
        total_elems = out_bytes / elem_size;

        out_ndims = vol_full_dims(ctx, out_dims);
 
        plane = 1;
        for (size_t i = 1; i < ctx->ndims; i++)
            plane *= ctx->dims[i];
        if (plane == 0) plane = 1;
    }
 
    if (chunk_elems > (uint64_t)total_elems)
        chunk_elems = (uint64_t)total_elems;

    if (chunk_elems % (uint64_t)plane != 0 ||
        (total_elems / plane) % (chunk_elems / plane) != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio-chunked container for '%s' records chunk_elems=%llu, "
                "which is not a whole number of %zu-element slices of the "
                "recorded shape (container written by a different VOL version, "
                "or _VOL_ORIG_DIMS disagrees with the writer)",
                ctx->compressor_id, (unsigned long long)chunk_elems, plane);
        return -1;
    }
 
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS PRESSIO: id=%s csize=%zu out_bytes=%zu rank=%zu "
           "dims=[", ctx->compressor_id, csize, out_bytes, out_ndims);
    for (size_t i = 0; i < out_ndims; i++)
        printf("%zu%s", out_dims[i], (i + 1 < out_ndims) ? "," : "");
    printf("] chunk_elems=%llu\n", (unsigned long long)chunk_elems);
#endif
 
    w = vol_get_chunk_wrapper(ctx, chunk_elems);
    if (!w)
        return -1;      /* error already pushed */
 
    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims, "malloc");

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
    {
        int _gpu = vol_is_gpu_codec(ctx->compressor_id);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
        double _t0;
        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }
 
        _t0 = bench_now_ms();
        int _derr = pressio_compressor_decompress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;
 
        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
            float _ms = 0.f;
            cudaEventElapsedTime(&_ms, _ev0, _ev1);
            ctx->device_ms += (double)_ms;
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
    {
        double _t0 = bench_now_ms();
        int _derr = pressio_compressor_decompress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;
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

herr_t
H5VL_pass_through_ext_decompress_pressio(compression_ctx *ctx,
                                         const void *cbuf, size_t csize,
                                         uint64_t chunk_elems,
                                         void *out, size_t out_bytes)
{
    try {
        return vol_decompress_pressio_impl(ctx, cbuf, csize, chunk_elems,
                                           out, out_bytes);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio-chunked decompress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio-chunked decompress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

} /* extern C */