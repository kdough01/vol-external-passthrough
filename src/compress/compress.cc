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

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#define VOL_BUF_HOST   0   /* malloc */
#define VOL_BUF_PINNED 1   /* cudaMallocHost -- page-locked, fast DMA */
#define VOL_BUF_DEVICE 2   /* cudaMalloc */

static int
vol_pool_log(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VOL_COMP_POOL_LOG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}

static size_t
vol_env_mb(const char *name, size_t dflt_mb)
{
    const char *e = getenv(name);
    if (e && *e) {
        char *end = NULL;
        unsigned long long v = strtoull(e, &end, 10);
        if (end != e) return (size_t)v * 1024u * 1024u;
    }
    return dflt_mb * 1024u * 1024u;
}

/* Page-locking multiple gigabytes starves the node and cudaMallocHost itself
 * gets slow enough to defeat the purpose, so oversized pinned requests are
 * quietly served as pageable host memory. */
static size_t
vol_pin_max(void)
{
    static size_t v = 0;
    static int    init = 0;
    if (!init) { init = 1; v = vol_env_mb("VOL_COMP_PIN_MAX_MB", 256); }
    return v;
}

static void *
vol_raw_alloc(size_t n, int kind)
{
    void *p = NULL;
#ifdef USE_CUDA
    if (kind == VOL_BUF_PINNED) {
        if (cudaMallocHost(&p, n) != cudaSuccess) { cudaGetLastError(); return NULL; }
        return p;
    }
    if (kind == VOL_BUF_DEVICE) {
        if (cudaMalloc(&p, n) != cudaSuccess) { cudaGetLastError(); return NULL; }
        return p;
    }
#else
    if (kind == VOL_BUF_DEVICE) return NULL;   /* never handed one out */
#endif
    return malloc(n);
}

static void
vol_raw_free(void *p, int kind)
{
    if (!p) return;
#ifdef USE_CUDA
    if (kind == VOL_BUF_PINNED) { cudaFreeHost(p); cudaGetLastError(); return; }
    if (kind == VOL_BUF_DEVICE) { cudaFree(p);     cudaGetLastError(); return; }
#endif
    free(p);
}

/* Geometric growth so a run that creeps upward does not realloc every write.
 * Floor at 1 MiB; below that the bookkeeping costs more than the memory. */
static size_t
vol_next_cap(size_t cur, size_t want)
{
    size_t n = cur ? cur : (size_t)(1u << 20);
    while (n < want) {
        size_t next = n + n / 2;
        if (next <= n) return want;          /* overflow guard */
        n = next;
    }
    return n;
}

static void
vol_buf_release(vol_buf_t *b)
{
    if (!b) return;
    vol_raw_free(b->ptr, b->kind);
    memset(b, 0, sizeof(*b));
}

/* Ensure b holds at least `want` bytes of `kind` memory; return b->ptr, or
 * NULL on failure (b is released). Contents are NOT preserved across a grow
 * -- these are scratch buffers. Never shrinks. */
static void *
vol_buf_reserve(vol_buf_t *b, size_t want, int kind)
{
    size_t new_cap;
    void  *np;

    if (!b) return NULL;
    if (want == 0) want = 1;

    /* Sticky downgrade: once a buffer has outgrown the pin cap, keep it
     * pageable. Oscillating around the cap would thrash the allocator on every
     * write, and cudaMallocHost is far more expensive than the copy it saves. */
    if (kind == VOL_BUF_PINNED &&
        (want > vol_pin_max() || (b->ptr && b->kind == VOL_BUF_HOST)))
        kind = VOL_BUF_HOST;

    /* Fast path. This is the whole point of the pool. */
    if (b->ptr && b->kind == kind && b->cap >= want)
        return b->ptr;

    /* Kind change: cannot realloc across allocators, so start over. */
    if (b->ptr && b->kind != kind) {
        vol_raw_free(b->ptr, b->kind);
        b->ptr = NULL;
        b->cap = 0;
    }

    new_cap = vol_next_cap(b->cap, want);
    np      = vol_raw_alloc(new_cap, kind);

    /* Geometric growth can overshoot into an allocation that will not fit.
     * Retry at the exact size before giving up. */
    if (!np && new_cap > want) {
        new_cap = want;
        np      = vol_raw_alloc(new_cap, kind);
    }
    if (!np) { vol_buf_release(b); return NULL; }

    vol_raw_free(b->ptr, b->kind);
    b->ptr  = np;
    b->cap  = new_cap;
    b->kind = kind;
    b->grows++;

    if (vol_pool_log())
        fprintf(stderr, "[pool] grow -> %.2f MiB (kind=%d, grow #%lu)\n",
                (double)new_cap / (1024.0 * 1024.0), kind, b->grows);
    return b->ptr;
}

static void
vol_bufpool_init(void)
{
    static int done = 0;
    if (done) return;
    done = 1;

#if defined(__GLIBC__)
    {
        const char *e = getenv("VOL_COMP_MALLOC_TUNE");
        if (e && *e == '0') {
            if (vol_pool_log())
                fprintf(stderr, "[pool] glibc tuning disabled\n");
            return;
        }

        mallopt(M_MMAP_MAX, 0);
        mallopt(M_TRIM_THRESHOLD, -1);

        if (vol_pool_log())
            fprintf(stderr, "[pool] glibc tuned: mmap off for malloc, "
                            "trim disabled\n");
    }
#endif
}

/* Helpers */
extern "C" int H5VL_pass_through_ext_buf_is_device(const void *p);
extern "C" int H5VL_pass_through_ext_compressor_available(const char *compressor_id);
extern "C" void *
H5VL_pass_through_ext_reserve_arena(compression_ctx *ctx, size_t nbytes)
{
    return vol_buf_reserve(&ctx->chunk_arena, nbytes, VOL_BUF_HOST);
}

void *
H5VL_pass_through_ext_reserve_chunk_hdr(compression_ctx *ctx, size_t nbytes)
{
    if (!ctx) return NULL;
    vol_bufpool_init();
    return vol_buf_reserve(&ctx->chunk_arena, nbytes, VOL_BUF_HOST);
}

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
vol_codec_is_gpu(const compression_ctx *ctx)
{
    return ctx && ctx->observed_gpu > 0;
}

/* Latch the observation. Idempotent: only the first call decides. */
static void
vol_note_output_domain(compression_ctx *ctx, const void *p)
{
    int dev;

    if (!ctx || ctx->observed_gpu >= 0)
        return;

    dev = H5VL_pass_through_ext_buf_is_device(p);
    ctx->observed_gpu = dev;

    if (getenv("VOL_COMP_COPY_LOG"))
        fprintf(stderr, "[VOL probe] '%s' produces %s-resident output; "
                        "device fast path %s\n",
                ctx->compressor_id, dev ? "device" : "host",
                dev ? "ENABLED" : "disabled");
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

static int
vol_force_host_output(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VOL_COMP_HOST_OUTPUT");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
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

static int
vol_time_h2d(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VOL_COMP_TIME_H2D");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}

static void
vol_make_device_readable(struct pressio_data *data)
{
    if (!data) return;
    if (strcmp(pressio_data_domain_id(data), "cudamalloc") != 0) {
        pressio_data *d = data;
        *d = domain_manager().make_readable(
            libpressio::domain_plugins().build("cudamalloc"), std::move(*d));
    }
}

static int
vol_stage_input_to_device(compression_ctx *ctx, struct pressio_data *input,
                          const void *src)
{
    double t0;

    if (!vol_time_h2d() || !ctx || !input)      return 0;
    if (H5VL_pass_through_ext_buf_is_device(src)) return 0;

    t0 = bench_now_ms();
    vol_make_device_readable(input);
    /* The migration may be issued async. Without a hard barrier the clock
     * closes before the DMA finishes and h2d_ms reads near zero. */
    cudaDeviceSynchronize();
    ctx->h2d_ms += bench_now_ms() - t0;

    if (getenv("VOL_COMP_COPY_LOG"))
        fprintf(stderr, "[h2d] '%s' staged input to device, cumulative %.3f ms\n",
                ctx->compressor_id, ctx->h2d_ms);
    return 1;
}
#endif

/* Allocate a codec output buffer in the domain where the codec will actually
 * write it.
 */
static struct pressio_data *
vol_new_output(enum pressio_dtype dt, size_t ndims, size_t *dims, int is_gpu)
{
#ifdef USE_CUDA
    if (is_gpu && !vol_force_host_output()) {
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

    if (getenv("VOL_COMP_VERIFY_EVENTS")) {
        double b0 = bench_now_ms();
        cudaError_t ce = ctx->cuda_stream_ok
                           ? cudaStreamSynchronize((cudaStream_t)ctx->stream)
                           : cudaDeviceSynchronize();
        double wait_ms = bench_now_ms() - b0;
        if (ce == cudaSuccess && wait_ms > 1.0 && (double)ms < 0.5 * wait_ms)
            fprintf(stderr,
                "[VOL events] '%s': event window %.2f ms but %.2f ms of device "
                "work was still outstanding afterwards. The codec is probably "
                "using a NON-BLOCKING stream, so legacy-default-stream event "
                "ordering does not apply and device_ms is UNDER-REPORTING. "
                "Confirm with nsys before using these numbers.\n",
                ctx->compressor_id, (double)ms, wait_ms);
    }
}

/* Does this codec declare a CUDA stream option? Returns 1 if the key exists
 * in any state, 0 if the codec has never heard of it. */
static int
vol_codec_declares_stream(compression_ctx *ctx, char *skey_out, size_t n)
{
    snprintf(skey_out, n, "%s:cuda_stream", ctx->compressor_id);

    struct pressio_options *decl = pressio_compressor_get_options(ctx->compressor);
    if (!decl) return 0;

    void *probe = NULL;
    enum pressio_options_key_status st =
        pressio_options_get_userptr(decl, skey_out, &probe);
    pressio_options_free(decl);

    /* key_set(0) and key_exists(1) both mean "known"; key_does_not_exist(2)
     * is what cuszp returns -- that is the status=2 in the warning. */
    return (st != pressio_options_key_does_not_exist);
}

/* Hand the codec our CUDA stream. */
static void
vol_set_cuda_stream(compression_ctx *ctx)
{
    char skey[128];

    if (ctx->cuda_stream_set) return;
    ctx->cuda_stream_set = 1;
    ctx->cuda_stream_ok  = 0;

    if (!vol_codec_declares_stream(ctx, skey, sizeof(skey))) {
        ctx->stream = NULL;
        if (getenv("VOL_COMP_COPY_LOG"))
            fprintf(stderr,
                "[VOL stream] '%s' declares no '%s' -- codec manages its own "
                "stream. Using the default stream + device barrier. device_ms "
                "is NOT reported for this codec, but pressio_call_ms IS now "
                "device-synchronized and therefore trustworthy.\n",
                ctx->compressor_id, skey);
        return;
    }

    /* Codec can take a stream, so give it a real one instead of NULL. */
    if (!ctx->stream) {
        cudaStream_t s = NULL;
        if (cudaStreamCreate(&s) != cudaSuccess) {
            fprintf(stderr, "[VOL stream] cudaStreamCreate failed for '%s'; "
                            "falling back to the default stream\n",
                    ctx->compressor_id);
            ctx->stream = NULL;
            return;
        }
        ctx->stream = (void *)s;
    }

    {
        struct pressio_options *sopt = pressio_options_new();
        pressio_options_set_userptr(sopt, skey, ctx->stream);
        int serr = pressio_compressor_set_options(ctx->compressor, sopt);
        pressio_options_free(sopt);
        if (serr)
            fprintf(stderr, "[VOL stream] set_options('%s') returned %d: %s\n",
                    skey, serr, pressio_compressor_error_msg(ctx->compressor));
    }

    {
        struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
        void *rb = NULL;
        enum pressio_options_key_status st =
            o ? pressio_options_get_userptr(o, skey, &rb)
              : pressio_options_key_does_not_exist;

        if (st == pressio_options_key_set && rb == ctx->stream) {
            ctx->cuda_stream_ok = 1;
            if (getenv("VOL_COMP_COPY_LOG"))
                fprintf(stderr, "[VOL stream] confirmed '%s' = %p\n",
                        skey, (void *)ctx->stream);
        } else {
            ctx->cuda_stream_ok = 0;
            fprintf(stderr,
                "[VOL stream] '%s' declares '%s' but did not accept our value "
                "(status=%d readback=%p expected=%p). Falling back to a device "
                "barrier; device_ms will not be reported.\n",
                ctx->compressor_id, skey, (int)st, rb, (void *)ctx->stream);
        }
        if (o) pressio_options_free(o);
    }
}
#endif /* USE_CUDA */

#ifdef USE_CUDA

static cudaError_t
vol_stream_barrier(compression_ctx *ctx, hid_t minor)
{
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "CUDA error pending before synchronization for '%s' "
                "(raised by earlier async work, likely inside the codec): %s",
                ctx->compressor_id, cudaGetErrorString(cerr));
        return cerr;
    }

    cerr = ctx->cuda_stream_ok
             ? cudaStreamSynchronize((cudaStream_t)ctx->stream)
             : cudaDeviceSynchronize();

    if (cerr != cudaSuccess)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, minor,
                "CUDA synchronization failed for '%s': %s",
                ctx->compressor_id, cudaGetErrorString(cerr));
    return cerr;
}
#endif

static int
vol_is_malloc_domain(const struct pressio_data *d)
{
    const char *dom = pressio_data_domain_id((struct pressio_data *)d);
    return dom && strcmp(dom, "malloc") == 0;
}

static struct pressio_data *
vol_reserve_comp_output(compression_ctx *ctx, size_t cap, int is_gpu,
                        size_t hdr_reserve, void **out_base)
{
    size_t dims[1];
    void  *base;

    *out_base = NULL;
    dims[0]   = cap;

    if (ctx->codec_ignores_output)
        return pressio_data_new_empty(pressio_byte_dtype, 0, NULL);

#ifdef USE_CUDA
    if (is_gpu && !vol_force_host_output()) {
        /* libpressio ALLOCATES AND OWNS the device output. Handing it a
         * nonowning cudamalloc buffer inverts the ownership contract: plugins
         * that call make_writeable() on their output free the pointer we are
         * still holding, and the pool then hands out a dangling device
         * pointer on the next write. */
        struct pressio_data *d =
            pressio_data_new_empty(pressio_byte_dtype, 1, dims);
        if (d) vol_make_device_resident(d);
        return d;
    }
#endif

    base = vol_buf_reserve(&ctx->comp_out, hdr_reserve + cap, VOL_BUF_HOST);
    if (!base) return NULL;

    *out_base = base;
    return pressio_data_new_nonowning_domain(pressio_byte_dtype,
                                             (char *)base + hdr_reserve,
                                             1, dims, "malloc");
}

/* Bring the codec's compressed output into memory this ctx owns, and report
 * where it landed. NOTHING here allocates a copy unless it has to.
 *
 *   CASE 1  codec wrote into the pooled buffer we gave it   -> zero copies
 *   CASE 2  codec allocated its own host buffer             -> ADOPT it, zero copies
 *   CASE 3  result is on the device, or unadoptable         -> one copy
 *
 * Case 2 is the common one: noop, sz3 and most libpressio codecs allocate
 * their own output regardless of what you pass in.
 *
 * Case 3 is the only irreducible per-byte cost, and it is timed into
 * ctx->transfer_ms so the overhead metric can exclude it.
 */
static int
vol_fetch_result_pooled(compression_ctx *ctx, struct pressio_data *result,
                        void *own_base, size_t hdr_reserve,
                        void **out_base, size_t *out_size)
{
    size_t sz  = 0;
    void  *src = pressio_data_ptr(result, &sz);
    void  *dst = NULL;

    *out_base = NULL;
    *out_size = 0;
    if (!src || sz == 0) return -1;

    vol_note_output_domain(ctx, src);

    /* ---- CASE 1: codec wrote into the pooled buffer we gave it ---- */
    if (own_base && src == (void *)((char *)own_base + hdr_reserve)) {
        *out_base = own_base;
        *out_size = sz;
        return 0;
    }

    /* ---- Device result: let libpressio bring it home ----
     * domain_manager knows the source domain, selects the transport, and
     * orders against the codec's stream. Do not hand-roll cudaMemcpyAsync
     * here -- we cannot order against a stream we were never given. */
    if (H5VL_pass_through_ext_buf_is_device(src)) {
        double _t0 = bench_now_ms();
        vol_make_host_resident(result);
        ctx->transfer_ms += bench_now_ms() - _t0;

        src = pressio_data_ptr(result, &sz);   /* it MOVED -- re-read both */
        if (!src || sz == 0) return -1;
    }

    /* ---- CASE 2: adopt the host buffer ----
     * Reached both by CPU codecs that allocate their own output and by GPU
     * codecs whose result we just migrated. Either way, zero further copies. */
    if (hdr_reserve == 0 && vol_is_malloc_domain(result)) {
        (void)static_cast<pressio_data *>(result)->release();

        free(ctx->adopted_buf);          /* the one from the PREVIOUS write */
        ctx->adopted_buf = src;
        ctx->codec_ignores_output = 1;

        if (getenv("VOL_COMP_COPY_LOG"))
            fprintf(stderr, "[copy] '%s' %zu B output adopted (no copy)\n",
                    ctx->compressor_id, sz);

        *out_base = src;
        *out_size = sz;
        return 0;
    }

    /* ---- CASE 3: unadoptable host buffer. Should be rare. ---- */
    if (getenv("VOL_COMP_COPY_LOG"))
        fprintf(stderr, "[copy] '%s' UNADOPTABLE host output, %zu B copy "
                        "(hdr_reserve=%zu domain=%s)\n",
                ctx->compressor_id, sz, hdr_reserve,
                pressio_data_domain_id(result));

    dst = vol_buf_reserve(&ctx->comp_stage, hdr_reserve + sz, VOL_BUF_HOST);
    if (!dst) return -2;
    {
        double _t0 = bench_now_ms();
        memcpy((char *)dst + hdr_reserve, src, sz);
        ctx->transfer_ms += bench_now_ms() - _t0;
    }

    *out_base = dst;
    *out_size = sz;
    return 0;
}

/* Fetch a compressed result the CALLER owns and frees.
 *
 * Distinct from vol_fetch_result_pooled because ctx->adopted_buf is a single
 * slot and the chunked path holds N results live at once.
 *
 * Zero-copy for host codecs: we steal the codec's allocation. One copy for
 * device results, which is unavoidable.
 */
static int
vol_fetch_result_owned(compression_ctx *ctx, struct pressio_data *result,
                       void **out_base, size_t *out_size)
{
    size_t sz  = 0;
    void  *src = pressio_data_ptr(result, &sz);

    *out_base = NULL;
    *out_size = 0;

    if (!src || sz == 0) return -1;

    vol_note_output_domain(ctx, src);

    /* Device result: let libpressio bring it home. The domain manager knows
     * the source domain, selects the transport, and orders against the
     * codec's own stream -- which is exactly what a hand-rolled
     * cudaMemcpyAsync on ctx->stream cannot do. */
    if (H5VL_pass_through_ext_buf_is_device(src)) {
        double _t0 = bench_now_ms();
        vol_make_host_resident(result);
        ctx->transfer_ms += bench_now_ms() - _t0;

        src = pressio_data_ptr(result, &sz);   /* it MOVED -- re-read both */
        if (!src || sz == 0) return -1;
    }

    /* Host-resident either way now. Steal it if the caller can free() it. */
    if (vol_is_malloc_domain(result)) {
        (void)static_cast<pressio_data *>(result)->release();
        *out_base = src;
        *out_size = sz;
        return 0;
    }

    {
        void *dst = malloc(sz);
        if (!dst) return -2;
        memcpy(dst, src, sz);
        *out_base = dst;
        *out_size = sz;
        return 0;
    }
}

/* Move the codec's output into the caller's buffer. 
Returns immediately when the codec already wrote
 * in place, which is the common case now that we hand it the caller's
 * buffer directly. */
static int
vol_fetch_into(compression_ctx *ctx, struct pressio_data *result,
               void *dst, size_t want)
{
    size_t sz  = 0;
    void  *src = pressio_data_ptr(result, &sz);

    if (!src || sz == 0) return -1;
    if (sz != want)      return -2;

    /* Decompressed straight into the caller's buffer. Nothing to do. */
    if (src == dst) return 0;

#ifdef USE_CUDA
    {
        int src_dev = H5VL_pass_through_ext_buf_is_device(src);
        int dst_dev = H5VL_pass_through_ext_buf_is_device(dst);

        if (src_dev || dst_dev) {
            enum cudaMemcpyKind kind =
                  (src_dev && dst_dev) ? cudaMemcpyDeviceToDevice
                : (src_dev)            ? cudaMemcpyDeviceToHost
                                       : cudaMemcpyHostToDevice;

            cudaError_t cerr = cudaMemcpyAsync(dst, src, want, kind,
                                               (cudaStream_t)ctx->stream);
            if (cerr == cudaSuccess)
                cerr = vol_stream_barrier(ctx, min_decompress_failed);
            if (cerr != cudaSuccess) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "copy of %zu decompressed bytes failed for '%s': %s",
                        want, ctx->compressor_id, cudaGetErrorString(cerr));
                return -3;
            }
            return 0;
        }
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

        {
            unsigned nthreads = 1;
            const char *nt = getenv("VOL_COMP_PRESSIO_NTHREADS");
            if (nt && *nt) {
                unsigned long v = strtoul(nt, NULL, 10);
                if (v > 0) nthreads = (unsigned)v;
            }
            pressio_options_set_uinteger(opts,
                "many_independent_threaded:nthreads", nthreads);
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

void
H5VL_pass_through_ext_release_buffers(compression_ctx *ctx)
{
    if (!ctx) return;

    if (vol_pool_log())
        fprintf(stderr, "[pool] '%s' final: comp_out=%.1f MiB/%lu grows, "
                        "stage=%.1f/%lu, hdr=%.1f/%lu, cont=%.1f/%lu, "
                        "decomp=%.1f/%lu\n",
                ctx->compressor_id ? ctx->compressor_id : "?",
                ctx->comp_out.cap    / 1048576.0, ctx->comp_out.grows,
                ctx->comp_stage.cap  / 1048576.0, ctx->comp_stage.grows,
                ctx->chunk_arena.cap / 1048576.0, ctx->chunk_arena.grows,
                ctx->cont_in.cap     / 1048576.0, ctx->cont_in.grows,
                ctx->decomp.cap      / 1048576.0, ctx->decomp.grows);

    free(ctx->adopted_buf);          /* the only owning pointer here */
    ctx->adopted_buf = NULL;

    vol_buf_release(&ctx->comp_out);
    vol_buf_release(&ctx->comp_stage);
    vol_buf_release(&ctx->chunk_arena);
    vol_buf_release(&ctx->cont_in);
    vol_buf_release(&ctx->decomp);

    /* These were views into the pools, not allocations. */
    ctx->decomp_buf            = NULL;
    ctx->decomp_size           = 0;
    ctx->read_served           = 0;
    ctx->compressed_buf        = NULL;
    ctx->compressed_chunk_size = 0;
}

void *
H5VL_pass_through_ext_reserve_decomp(compression_ctx *ctx, size_t nbytes)
{
    if (!ctx) return NULL;
    vol_bufpool_init();
    return vol_buf_reserve(&ctx->decomp, nbytes, VOL_BUF_HOST);
}

void *
H5VL_pass_through_ext_reserve_container(compression_ctx *ctx, size_t nbytes)
{
    if (!ctx) return NULL;
    vol_bufpool_init();
    return vol_buf_reserve(&ctx->cont_in, nbytes, VOL_BUF_HOST);
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

/* ------------------------------------------------------------------------
 * One-time CUDA warmup.
 * Called from the compress/decompress entry points before any timer starts.
 * --------------------------------------------------------------------- */
void
H5VL_pass_through_ext_cuda_warmup(void)
{
    vol_bufpool_init();
#ifdef USE_CUDA
    static int done = 0;
    void *p = NULL;

    if (done) return;
    done = 1;

    if (getenv("VOL_COMP_NO_WARMUP")) return;

    if (cudaMalloc(&p, 1024) == cudaSuccess) {
        cudaMemset(p, 0, 1024);
        cudaDeviceSynchronize();
        cudaFree(p);
    }
    cudaGetLastError();     /* do not leak a warmup failure into real work */

    if (getenv("VOL_COMP_COPY_LOG"))
        fprintf(stderr, "[VOL warmup] CUDA context initialized before timing\n");
#endif
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
    size_t  rdims[H5S_MAX_RANK];
    size_t  logical;
    size_t  cap      = 0;
    void   *own_base = NULL;
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
        for (size_t i = 0; i < ctx->ndims; i++)
            rdims[i] = ctx->dims[ctx->ndims - 1 - i];
        in_dims = rdims;
    }

    is_gpu = vol_codec_is_gpu(ctx);

    if (getenv("VOL_COMP_DUMP_CALL")) {
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

    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              in_ndims, in_dims,
                                              vol_ptr_domain(data));

    cap    = nbytes + nbytes / 8 + (1u << 16);
    output = vol_reserve_comp_output(ctx, cap, is_gpu, hdr_reserve, &own_base);

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to reserve pressio buffers for a %zu byte dataset", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    (void)vol_stage_input_to_device(ctx, input, data);
    vol_set_cuda_stream(ctx);
    ev = vol_ev_begin(ctx, 1);
#endif
    t0   = bench_now_ms();
    cerr = pressio_compressor_compress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* HOST time only */
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

    frc = vol_fetch_result_pooled(ctx, output, own_base, hdr_reserve,
                                  &base, &csize);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compress of '%s' produced no usable output for %zu B "
                "(fetch rc=%d)", ctx->compressor_id, nbytes, frc);
        ret_val = -1;
        goto done;
    }

    /* Truncation check only applies when WE supplied the buffer. If the codec
     * allocated its own it sized it itself, and a legitimate expansion beyond
     * cap is not a truncation. */
    if (own_base && csize >= cap) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' filled the entire %zu byte output buffer for "
                "%zu bytes of input -- the result is truncated and would decode "
                "to garbage", ctx->compressor_id, cap, nbytes);
        ret_val = -1;
        goto done;
    }
    if (csize >= nbytes)
        fprintf(stderr,
            "[VOL WARN] '%s' EXPANDED %zu bytes to %zu bytes (%.4f%% of input).\n",
            ctx->compressor_id, nbytes, csize,
            100.0 * (double)csize / (double)nbytes);

    *out_cbuf  = base;      /* ctx owns this: pooled or adopted. Do NOT free. */
    *out_csize = (uint64_t)csize;

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
        H5VL_pass_through_ext_cuda_warmup();
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
#ifdef USE_CUDA
        if (H5VL_pass_through_ext_buf_is_device(out)) {
            cudaError_t cerr = cudaMemcpyAsync(out, cbuf, n,
                                               cudaMemcpyHostToDevice,
                                               (cudaStream_t)ctx->stream);
            if (cerr == cudaSuccess)
                cerr = vol_stream_barrier(ctx, min_decompress_failed);
            if (cerr != cudaSuccess) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "host-to-device copy of %zu bytes failed: %s",
                        n, cudaGetErrorString(cerr));
                return -1;
            }
            return 0;
        }
#endif
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
        for (size_t i = 0; i < ctx->ndims; i++)
            rdims[i] = ctx->dims[ctx->ndims - 1 - i];
        out_dims = rdims;
    }

    is_gpu = vol_codec_is_gpu(ctx);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS NATIVE: id=%s csize=%zu out_bytes=%zu ndims=%zu\n",
           ctx->compressor_id, csize, out_bytes, out_ndims);
#endif

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims,
                                              vol_ptr_domain(cbuf));

    output = pressio_data_new_nonowning_domain(out_dtype, out,
                                               out_ndims, out_dims,
                                               vol_ptr_domain(out));
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
    ev = vol_ev_begin(ctx, 1);
#endif
    t0   = bench_now_ms();
    derr = pressio_compressor_decompress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* HOST time only */
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
        H5VL_pass_through_ext_cuda_warmup();
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
    size_t out_dims[1];
    size_t in_ndims;
    size_t cap    = 0;
    void  *base   = NULL;
    size_t csize  = 0;
    int    is_gpu = 0, ev = 0, cerr = 0, frc = 0;
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
        if (in_ndims == 0) return -1;          /* error already pushed */
    }

    is_gpu = vol_codec_is_gpu(ctx);
    cap         = nbytes + nbytes / 8 + (1u << 16);
    out_dims[0] = cap;

    input = pressio_data_new_nonowning_domain(in_dtype, (void *)data,
                                              in_ndims, chunk_dims,
                                              vol_ptr_domain(data));

#ifdef USE_CUDA
    if (is_gpu) {
        /* libpressio ALLOCATES AND OWNS the device output. Handing it a
         * nonowning cudamalloc buffer inverts the ownership contract: plugins
         * that call make_writeable() on their output free the pointer we are
         * still holding, and we would then hand out a dangling device pointer
         * on the next chunk. */
        output = pressio_data_new_empty(pressio_byte_dtype, 1, out_dims);
        if (output) vol_make_device_resident(output);
    } else
#endif
    {
        /* Host codecs mostly ignore this and allocate their own, which
         * vol_fetch_result_owned then steals for free. Once that is known,
         * stop reserving a buffer nobody touches. */
        output = ctx->codec_ignores_output
                   ? pressio_data_new_empty(pressio_byte_dtype, 0, NULL)
                   : pressio_data_new_owning(pressio_byte_dtype, 1, out_dims);
    }

    if (!input || !output) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to allocate pressio buffers for a %zu byte chunk", nbytes);
        ret_val = -1;
        goto done;
    }

#ifdef USE_CUDA
    (void)vol_stage_input_to_device(ctx, input, data);
    vol_set_cuda_stream(ctx);
    ev = vol_ev_begin(ctx, 1);
#endif
    t0   = bench_now_ms();
    cerr = pressio_compressor_compress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* HOST time, per chunk */
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

    /* Migrates a device result host-side via the domain manager, then steals
     * the host buffer. Caller owns and frees what comes back. */
    frc = vol_fetch_result_owned(ctx, output, &base, &csize);
    if (frc != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' produced no usable output for a %zu byte chunk "
                "(fetch rc=%d)", ctx->compressor_id, nbytes, frc);
        ret_val = -1;
        goto done;
    }

    /* Truncation check only applies when WE supplied a fixed-size buffer. If
     * the codec allocated its own it sized it itself, and a legitimate
     * expansion past cap is not a truncation. */
    if (!ctx->codec_ignores_output && !is_gpu && csize >= cap) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "compressor '%s' filled the entire %zu byte output buffer for a "
                "%zu byte chunk -- the result is truncated and would decode to "
                "garbage. Increase the headroom in "
                "vol_transfer_compress_chunk_impl",
                ctx->compressor_id, cap, nbytes);
        free(base);
        ret_val = -1;
        goto done;
    }

    if (csize >= nbytes)
        fprintf(stderr,
            "[VOL WARN] '%s' EXPANDED a %zu byte chunk to %zu bytes (%.4f%% of "
            "input). Headroom was %zu bytes.\n",
            ctx->compressor_id, nbytes, csize,
            100.0 * (double)csize / (double)nbytes, cap - nbytes);
    else if (getenv("VOL_COMP_CHUNK_LOG"))
        fprintf(stderr, "[chunk-out] '%s' rank=%zu %zu -> %zu B (%.2fx), cap %zu B\n",
                ctx->compressor_id, in_ndims, nbytes, csize,
                (double)nbytes / (double)csize, cap);

    *out_cbuf  = base;      /* CALLER OWNS AND FREES THIS */
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
        H5VL_pass_through_ext_cuda_warmup();
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
#ifdef USE_CUDA
        if (H5VL_pass_through_ext_buf_is_device(out)) {
            cudaError_t cerr = cudaMemcpyAsync(out, cbuf, n,
                                               cudaMemcpyHostToDevice,
                                               (cudaStream_t)ctx->stream);
            if (cerr == cudaSuccess)
                cerr = vol_stream_barrier(ctx, min_decompress_failed);
            if (cerr != cudaSuccess) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "host-to-device copy of %zu bytes failed: %s",
                        n, cudaGetErrorString(cerr));
                return -1;
            }
            return 0;
        }
#endif
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

    is_gpu = vol_codec_is_gpu(ctx);

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- DECOMPRESS CHUNK: id=%s csize=%zu out_bytes=%zu rank=%zu "
           "dims=[", ctx->compressor_id, csize, out_bytes, out_ndims);
    for (size_t i = 0; i < out_ndims; i++)
        printf("%zu%s", chunk_dims[i], (i + 1 < out_ndims) ? "," : "");
    printf("]\n");
#endif

    comp_dims[0] = csize;
    input = pressio_data_new_nonowning_domain(pressio_byte_dtype, (void *)cbuf,
                                              1, comp_dims,
                                              vol_ptr_domain(cbuf));

    output = pressio_data_new_nonowning_domain(out_dtype, out,
                                               out_ndims, chunk_dims,
                                               vol_ptr_domain(out));
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
    ev = vol_ev_begin(ctx, 1);
#endif
    t0   = bench_now_ms();
    derr = pressio_compressor_decompress(ctx->compressor, input, output);
    ctx->pressio_call_ms += bench_now_ms() - t0;   /* HOST time, per chunk */
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
        H5VL_pass_through_ext_cuda_warmup();
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
    ctx->h2d_ms          = 0.0;
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
        if (strcasecmp(env, "shared") == 0)
            return VOL_CHUNKING_SHARED;
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
        else if (strcasecmp(mode, "shared") == 0)
            ctx->chunking_mode = VOL_CHUNKING_SHARED;
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
                          size_t chunk_bytes_req, size_t hdr_reserve,
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

    /* The chunking wrapper always allocates its own output, so there is no
     * point handing it a pooled buffer -- vol_fetch_result_pooled will copy
     * the result into the pool afterwards. */
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
        int _gpu = vol_codec_is_gpu(ctx);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
        double _t0;
        int _cerr;
        float _ms = 0.f;
        (void)vol_stage_input_to_device(ctx, input, data);

        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        _t0   = bench_now_ms();
        _cerr = pressio_compressor_compress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;   /* HOST time only */

        /* Device time comes from the events, not from the host clock. */
        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
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

    /* The old code called vol_make_host_resident(output) -- a full extra copy
     * of the blob -- and then malloc'd and memcpy'd it again. The pooled fetch
     * handles the device case itself and lands the result straight in the
     * pool, at hdr_reserve so the container header can be written in front. */
    {
        void  *base  = NULL;
        size_t csize = 0;

        if (vol_fetch_result_pooled(ctx, output, NULL, hdr_reserve,
                                    &base, &csize) != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "pressio-chunked compress of '%s' produced no usable "
                    "output for %zu B", ctx->compressor_id, nbytes);
            ret_val = -1;
            goto done;
        }

        *out_cbuf        = base;
        *out_csize       = (uint64_t)csize;
        *out_chunk_elems = chunk_elems;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("COMPRESS PRESSIO OK: id=%s comp_size=%zu original_nbytes=%zu "
               "chunk_elems=%llu\n",
               ctx->compressor_id, csize, nbytes,
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
                                       size_t hdr_reserve,
                                       void **out_cbuf, uint64_t *out_csize,
                                       uint64_t *out_chunk_elems)
{
    try {
        H5VL_pass_through_ext_cuda_warmup();
        return vol_compress_pressio_impl(ctx, data, nbytes, chunk_bytes_req,
                                         hdr_reserve, out_cbuf, out_csize,
                                         out_chunk_elems);
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

    output = pressio_data_new_nonowning_domain(out_dtype, out,
                                               out_ndims, out_dims,
                                               vol_ptr_domain(out));
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
        int _gpu = vol_codec_is_gpu(ctx);
        cudaEvent_t _ev0 = NULL, _ev1 = NULL;
        cudaStream_t _stream = (cudaStream_t)ctx->stream;   /* 0 => default */
        double _t0;
        int _derr;
        float _ms = 0.f;

        if (_gpu) {
            cudaEventCreate(&_ev0);
            cudaEventCreate(&_ev1);
            cudaEventRecord(_ev0, _stream);
        }

        _t0   = bench_now_ms();
        _derr = pressio_compressor_decompress(w, input, output);
        ctx->pressio_call_ms += bench_now_ms() - _t0;   /* HOST time only */

        if (_gpu) {
            cudaEventRecord(_ev1, _stream);
            cudaEventSynchronize(_ev1);
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

    {
        size_t actual_bytes = 0;
        void  *chk = pressio_data_ptr(output, &actual_bytes);

        if (!chk || actual_bytes == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "pressio-chunked decompress of '%s' produced no output",
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
    }

    if (vol_fetch_into(ctx, output, out, out_bytes) != 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio-chunked decompress of '%s' could not deliver %zu bytes",
                ctx->compressor_id, out_bytes);
        ret_val = -1;
        goto done;
    }

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("DECOMPRESS PRESSIO OK: id=%s out_bytes=%zu\n",
           ctx->compressor_id, out_bytes);
#endif

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
        H5VL_pass_through_ext_cuda_warmup();
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