#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

extern "C" {
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"
#include "vol_errors.h"
#include "vol_shared_meta.h"
}

#if defined(__has_include)
#  if __has_include(<zdict.h>) && __has_include(<zstd.h>)
#    ifndef VOL_HAVE_ZDICT
#      define VOL_HAVE_ZDICT 1
#    endif
#  endif
#endif

#if VOL_HAVE_ZDICT
#  include <zstd.h>
#  include <zdict.h>
#endif

/* Provided by the main compression translation unit. */
extern "C" herr_t H5VL_pass_through_ext_transfer_compress_chunk(
    compression_ctx *ctx, const void *data, size_t nbytes,
    void **out_cbuf, uint64_t *out_csize);
extern "C" herr_t H5VL_pass_through_ext_transfer_decompress_chunk(
    compression_ctx *ctx, const void *cbuf, size_t csize,
    void *out, size_t out_bytes);
extern "C" size_t vol_logical_nbytes(const compression_ctx *ctx);

namespace {

struct shared_state {
    const compression_ctx *ctx;
    void   *cdict;          /* ZSTD_CDict*  */
    void   *ddict;          /* ZSTD_DDict*  */
    void   *cctx;           /* ZSTD_CCtx*   */
    void   *dctx;           /* ZSTD_DCtx*   */
    void   *dict_buf;       /* raw dictionary bytes (host)  */
    size_t  dict_len;
    int     level;
};

#define VOL_SHARED_MAX_CTX 64
shared_state g_states[VOL_SHARED_MAX_CTX];
int          g_states_init = 0;

shared_state *state_find(const compression_ctx *ctx, int create)
{
    if (!g_states_init) {
        memset(g_states, 0, sizeof(g_states));
        g_states_init = 1;
    }
    for (int i = 0; i < VOL_SHARED_MAX_CTX; i++)
        if (g_states[i].ctx == ctx) return &g_states[i];
    if (!create) return NULL;
    for (int i = 0; i < VOL_SHARED_MAX_CTX; i++)
        if (g_states[i].ctx == NULL) {
            memset(&g_states[i], 0, sizeof(g_states[i]));
            g_states[i].ctx = ctx;
            return &g_states[i];
        }
    return NULL;
}

/* ---- little helpers -------------------------------------------------- */

inline void put64(unsigned char *p, size_t word, uint64_t v)
{
    memcpy(p + word * sizeof(uint64_t), &v, sizeof(uint64_t));
}
inline uint64_t get64(const unsigned char *p, size_t word)
{
    uint64_t v;
    memcpy(&v, p + word * sizeof(uint64_t), sizeof(uint64_t));
    return v;
}

} /* anonymous namespace */

#if VOL_HAVE_ZDICT

static int zstd_can_share(const compression_ctx *ctx)
{
    if (!ctx || !ctx->compressor_id) return 0;
    /* Only claim plain zstd. blosc's internal zstd is driven through blosc's
     * own options and is not ours to redirect. */
    return strcmp(ctx->compressor_id, "zstd") == 0;
}

/* Pull the requested level out of the codec options if present. */
static int zstd_level_of(compression_ctx *ctx)
{
    int lvl = 3;
    struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
    if (o) {
        int v = 0;
        if (pressio_options_get_integer(o, "zstd:level", &v) ==
                pressio_options_key_set && v != 0)
            lvl = v;
        pressio_options_free(o);
    }
    return lvl;
}

static int zstd_derive(compression_ctx *ctx, const void *data, size_t nbytes,
                       size_t chunk_bytes, void **out_meta, size_t *out_len)
{
    shared_state *st = state_find(ctx, 1);
    if (!st) return -1;

    *out_meta = NULL;
    *out_len  = 0;

    if (chunk_bytes == 0 || nbytes == 0) return 0;

    /* Sample across the WHOLE array so the dictionary reflects global
     * structure, not just the head. zstd's trainer wants many smallish
     * samples; cap the work so training stays negligible next to compression. */
    const size_t nchunks   = (nbytes + chunk_bytes - 1) / chunk_bytes;
    size_t       nsamples  = nchunks < 2048 ? nchunks : 2048;
    if (nsamples < 8) nsamples = (nchunks < 8) ? nchunks : 8;
    if (nsamples == 0) return 0;

    size_t sample_len = chunk_bytes < (size_t)(64 * 1024)
                            ? chunk_bytes : (size_t)(64 * 1024);
    if (sample_len == 0) return 0;

    /* Dictionary size: zstd's own guidance is ~110 KB, and it must stay small
     * relative to the data or it cannot pay for itself. */
    size_t dict_cap = 112640;
    if (dict_cap > nbytes / 100) dict_cap = nbytes / 100;
    if (dict_cap < 4096) dict_cap = 4096;
    if (dict_cap > nbytes / 2) return 0;         /* dataset too small to bother */

    unsigned char *samples = (unsigned char *)malloc(nsamples * sample_len);
    size_t        *sizes   = (size_t *)malloc(nsamples * sizeof(size_t));
    void          *dict    = malloc(dict_cap);
    if (!samples || !sizes || !dict) {
        free(samples); free(sizes); free(dict);
        return -1;
    }

    const size_t stride = nchunks / nsamples ? nchunks / nsamples : 1;
    for (size_t i = 0; i < nsamples; i++) {
        size_t coff = (i * stride) * chunk_bytes;
        if (coff + sample_len > nbytes)
            coff = (nbytes > sample_len) ? (nbytes - sample_len) : 0;
        size_t len = (coff + sample_len <= nbytes) ? sample_len : (nbytes - coff);
        memcpy(samples + i * sample_len, (const unsigned char *)data + coff, len);
        sizes[i] = len;
    }

    size_t dsz = ZDICT_trainFromBuffer(dict, dict_cap, samples,
                                       sizes, (unsigned)nsamples);
    free(samples);
    free(sizes);

    if (ZDICT_isError(dsz) || dsz == 0) {
        /* Not fatal: training legitimately fails on data with no cross-chunk
         * redundancy. Decline and let the generic path handle it. */
        if (getenv("VOL_COMP_CHUNK_LOG"))
            fprintf(stderr, "[shared] zstd dictionary training declined: %s\n",
                    ZDICT_isError(dsz) ? ZDICT_getErrorName(dsz) : "empty");
        free(dict);
        return 0;
    }

    st->level = zstd_level_of(ctx);
    st->dict_buf = dict;
    st->dict_len = dsz;
    st->cdict = ZSTD_createCDict(dict, dsz, st->level);
    st->cctx  = ZSTD_createCCtx();
    if (!st->cdict || !st->cctx) {
        if (st->cdict) ZSTD_freeCDict((ZSTD_CDict *)st->cdict);
        if (st->cctx)  ZSTD_freeCCtx((ZSTD_CCtx *)st->cctx);
        st->cdict = st->cctx = NULL;
        free(dict);
        st->dict_buf = NULL; st->dict_len = 0;
        return -1;
    }

    /* Caller takes ownership of a COPY; st->dict_buf stays for our own use. */
    void *copy = malloc(dsz);
    if (!copy) return -1;
    memcpy(copy, dict, dsz);
    *out_meta = copy;
    *out_len  = dsz;

    if (getenv("VOL_COMP_CHUNK_LOG"))
        fprintf(stderr, "[shared] trained %zu B zstd dictionary from %zu samples "
                        "of %zu B (level %d)\n",
                dsz, nsamples, sample_len, st->level);
    return 0;
}

static int zstd_attach(compression_ctx *ctx, const void *meta, size_t len)
{
    shared_state *st = state_find(ctx, 1);
    if (!st) return -1;
    if (!meta || len == 0) return 0;         /* container has no dictionary */

    st->ddict = ZSTD_createDDict(meta, len);
    st->dctx  = ZSTD_createDCtx();
    if (!st->ddict || !st->dctx) {
        if (st->ddict) ZSTD_freeDDict((ZSTD_DDict *)st->ddict);
        if (st->dctx)  ZSTD_freeDCtx((ZSTD_DCtx *)st->dctx);
        st->ddict = st->dctx = NULL;
        return -1;
    }
    return 0;
}

static int zstd_compress_chunk(compression_ctx *ctx, const void *in, size_t nbytes,
                               void **out_cbuf, uint64_t *out_csize)
{
    shared_state *st = state_find(ctx, 0);
    if (!st || !st->cdict || !st->cctx) return -1;

    size_t cap = ZSTD_compressBound(nbytes);
    void  *buf = malloc(cap);
    if (!buf) return -1;

    size_t got = ZSTD_compress_usingCDict((ZSTD_CCtx *)st->cctx, buf, cap,
                                          in, nbytes,
                                          (ZSTD_CDict *)st->cdict);
    if (ZSTD_isError(got) || got == 0) {
        free(buf);
        return -1;
    }
    *out_cbuf  = buf;
    *out_csize = (uint64_t)got;
    return 0;
}

static int zstd_decompress_chunk(compression_ctx *ctx, const void *cbuf, size_t csize,
                                 void *out, size_t out_bytes)
{
    shared_state *st = state_find(ctx, 0);
    if (!st || !st->ddict || !st->dctx) return -1;

    size_t got = ZSTD_decompress_usingDDict((ZSTD_DCtx *)st->dctx,
                                            out, out_bytes, cbuf, csize,
                                            (ZSTD_DDict *)st->ddict);
    if (ZSTD_isError(got) || got != out_bytes) return -1;
    return 0;
}

static void zstd_release(compression_ctx *ctx)
{
    shared_state *st = state_find(ctx, 0);
    if (!st) return;
    if (st->cdict) ZSTD_freeCDict((ZSTD_CDict *)st->cdict);
    if (st->ddict) ZSTD_freeDDict((ZSTD_DDict *)st->ddict);
    if (st->cctx)  ZSTD_freeCCtx((ZSTD_CCtx *)st->cctx);
    if (st->dctx)  ZSTD_freeDCtx((ZSTD_DCtx *)st->dctx);
    free(st->dict_buf);
    memset(st, 0, sizeof(*st));
}

static const vol_shared_provider g_zstd_provider = {
    "zstd-dict",
    zstd_can_share,
    zstd_derive,
    zstd_attach,
    zstd_compress_chunk,
    zstd_decompress_chunk,
    zstd_release
};
#endif /* VOL_HAVE_ZDICT */

/* =========================================================================
 * REGISTRY
 * ========================================================================= */
static const vol_shared_provider *g_providers[] = {
#if VOL_HAVE_ZDICT
    &g_zstd_provider,
#endif
    NULL
};

extern "C" const vol_shared_provider *
vol_shared_provider_for(const compression_ctx *ctx)
{
    if (!ctx) return NULL;
    if (getenv("VOL_SHARED_DISABLE")) return NULL;
    for (int i = 0; g_providers[i]; i++)
        if (g_providers[i]->can_share(ctx))
            return g_providers[i];
    return NULL;
}

extern "C" const char *
vol_shared_providers_list(void)
{
#if VOL_HAVE_ZDICT
    return "zstd-dict";
#else
    return "(none compiled in -- zdict.h/zstd.h not found at build time)";
#endif
}

/* =========================================================================
 * CONTAINER ASSEMBLY
 * ========================================================================= */
extern "C" herr_t
H5VL_pass_through_ext_shared_probe(const void *cbuf, size_t cont_bytes,
                                   uint64_t *out_nchunks,
                                   uint64_t *out_chunk_bytes,
                                   uint64_t *out_total_bytes,
                                   uint64_t *out_shared_len)
{
    const unsigned char *p = (const unsigned char *)cbuf;

    if (!cbuf || cont_bytes < VOL_SHARED_HDR_WORDS * sizeof(uint64_t))
        return -1;
    if (get64(p, 0) != VOL_SHARED_MAGIC) return -1;
    if (get64(p, 1) != VOL_SHARED_VERSION) return -1;

    const uint64_t nregions = get64(p, 2);
    if (out_nchunks)     *out_nchunks     = get64(p, 3);
    if (out_chunk_bytes) *out_chunk_bytes = get64(p, 4);
    if (out_total_bytes) *out_total_bytes = get64(p, 5);

    if (out_shared_len) {
        *out_shared_len = 0;
        for (uint64_t r = 0; r < nregions; r++) {
            size_t w = VOL_SHARED_HDR_WORDS + r * VOL_SHARED_DIR_WORDS;
            if ((w + VOL_SHARED_DIR_WORDS) * sizeof(uint64_t) > cont_bytes)
                return -1;
            if (get64(p, w) == VOL_REGION_SHARED_META) {
                *out_shared_len = get64(p, w + 3);
                break;
            }
        }
    }
    return 0;
}

static herr_t
vol_compress_shared_impl(compression_ctx *ctx,
                         const void *data, size_t nbytes,
                         size_t chunk_bytes_req,
                         void **out_container, uint64_t *out_total)
{
    const vol_shared_provider *prov = NULL;
    void      *shared_meta = NULL;
    size_t     shared_len  = 0;
    void     **cbufs       = NULL;
    uint64_t  *csizes      = NULL;
    size_t     nchunks     = 0;
    size_t     chunk_bytes = chunk_bytes_req;
    herr_t     ret         = 0;

    if (!ctx || !out_container || !out_total || !data || nbytes == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "invalid arguments to compress_shared");
        return -1;
    }
    *out_container = NULL;
    *out_total     = 0;

    if (chunk_bytes == 0 || chunk_bytes > nbytes) chunk_bytes = nbytes;
    nchunks = (nbytes + chunk_bytes - 1) / chunk_bytes;

    prov = vol_shared_provider_for(ctx);

    /* ---- 1. derive the shared metadata ONCE, before the chunk loop ---- */
    if (prov && prov->derive) {
        if (prov->derive(ctx, data, nbytes, chunk_bytes,
                         &shared_meta, &shared_len) < 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "shared-metadata provider '%s' failed to derive metadata "
                    "for '%s'", prov->name, ctx->compressor_id);
            return -1;
        }
        if (shared_len == 0) prov = NULL;      /* declined -> generic path */
    }

    cbufs  = (void **)calloc(nchunks, sizeof(void *));
    csizes = (uint64_t *)calloc(nchunks, sizeof(uint64_t));
    if (!cbufs || !csizes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "out of memory allocating chunk table (%zu chunks)", nchunks);
        free(cbufs); free(csizes); free(shared_meta);
        return -1;
    }

    /* ---- 2. compress each chunk AGAINST the shared metadata ---- */
    for (size_t k = 0; k < nchunks && ret == 0; k++) {
        const size_t coff = k * chunk_bytes;
        const size_t clen = (nbytes - coff < chunk_bytes)
                                ? (nbytes - coff) : chunk_bytes;
        int rc;

        if (prov)
            rc = prov->compress_chunk(ctx, (const unsigned char *)data + coff,
                                      clen, &cbufs[k], &csizes[k]);
        else
            rc = (int)H5VL_pass_through_ext_transfer_compress_chunk(
                     ctx, (const unsigned char *)data + coff, clen,
                     &cbufs[k], &csizes[k]);

        if (rc < 0 || cbufs[k] == NULL || csizes[k] == 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "shared-container compression failed at chunk %zu/%zu for "
                    "'%s'%s", k, nchunks, ctx->compressor_id,
                    prov ? " (provider path)" : " (generic path)");
            ret = -1;
        }
    }

    /* ---- 3. assemble ONE container: header + directory + regions ---- */
    if (ret == 0) {
        const uint64_t nregions = 3;   /* shared meta, chunk table, payload */
        const size_t hdr_bytes =
            (VOL_SHARED_HDR_WORDS + nregions * VOL_SHARED_DIR_WORDS)
            * sizeof(uint64_t);
        const size_t table_bytes = nchunks * sizeof(uint64_t);
        size_t payload_bytes = 0;
        for (size_t k = 0; k < nchunks; k++) payload_bytes += csizes[k];

        const size_t meta_off    = hdr_bytes;
        const size_t table_off   = meta_off + shared_len;
        const size_t payload_off = table_off + table_bytes;
        const size_t total       = payload_off + payload_bytes;

        unsigned char *c = (unsigned char *)malloc(total);
        if (!c) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory assembling a %zu byte shared container", total);
            ret = -1;
        } else {
            put64(c, 0, VOL_SHARED_MAGIC);
            put64(c, 1, VOL_SHARED_VERSION);
            put64(c, 2, nregions);
            put64(c, 3, (uint64_t)nchunks);
            put64(c, 4, (uint64_t)chunk_bytes);
            put64(c, 5, (uint64_t)nbytes);
            put64(c, 6, 0);
            put64(c, 7, 0);

            size_t w = VOL_SHARED_HDR_WORDS;
            put64(c, w + 0, VOL_REGION_SHARED_META);
            put64(c, w + 1, 0);
            put64(c, w + 2, (uint64_t)meta_off);
            put64(c, w + 3, (uint64_t)shared_len);
            w += VOL_SHARED_DIR_WORDS;
            put64(c, w + 0, VOL_REGION_CHUNK_TABLE);
            put64(c, w + 1, 0);
            put64(c, w + 2, (uint64_t)table_off);
            put64(c, w + 3, (uint64_t)table_bytes);
            w += VOL_SHARED_DIR_WORDS;
            put64(c, w + 0, VOL_REGION_PAYLOAD);
            put64(c, w + 1, 0);
            put64(c, w + 2, (uint64_t)payload_off);
            put64(c, w + 3, (uint64_t)payload_bytes);

            if (shared_len) memcpy(c + meta_off, shared_meta, shared_len);
            memcpy(c + table_off, csizes, table_bytes);

            size_t off = payload_off;
            for (size_t k = 0; k < nchunks; k++) {
                memcpy(c + off, cbufs[k], (size_t)csizes[k]);
                off += (size_t)csizes[k];
            }

            *out_container = c;
            *out_total     = (uint64_t)total;

            if (getenv("VOL_COMP_CHUNK_LOG"))
                fprintf(stderr,
                    "[shared] '%s' provider=%s nchunks=%zu chunk=%zu B "
                    "shared_meta=%zu B payload=%zu B total=%zu B (%.3fx)\n",
                    ctx->compressor_id, prov ? prov->name : "(none)",
                    nchunks, chunk_bytes, shared_len, payload_bytes, total,
                    (double)nbytes / (double)total);
        }
    }

    for (size_t k = 0; k < nchunks; k++) free(cbufs[k]);
    free(cbufs);
    free(csizes);
    free(shared_meta);
    if (prov && prov->release) prov->release(ctx);
    return ret;
}

extern "C" herr_t
H5VL_pass_through_ext_compress_shared(compression_ctx *ctx,
                                      const void *data, size_t nbytes,
                                      size_t chunk_bytes_req,
                                      void **out_container, uint64_t *out_total)
{
    try {
        return vol_compress_shared_impl(ctx, data, nbytes, chunk_bytes_req,
                                        out_container, out_total);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "shared compress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "shared compress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

static herr_t
vol_decompress_shared_impl(compression_ctx *ctx,
                           const void *cbuf, size_t cont_bytes,
                           void *out, size_t out_bytes)
{
    const unsigned char *p = (const unsigned char *)cbuf;
    const vol_shared_provider *prov = NULL;
    uint64_t nregions, nchunks, chunk_bytes, total_bytes;
    size_t   meta_off = 0, meta_len = 0;
    size_t   table_off = 0, table_len = 0;
    size_t   payload_off = 0, payload_len = 0;
    herr_t   ret = 0;

    if (!ctx || !cbuf || !out ||
        cont_bytes < VOL_SHARED_HDR_WORDS * sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid arguments to decompress_shared");
        return -1;
    }

    if (get64(p, 0) != VOL_SHARED_MAGIC || get64(p, 1) != VOL_SHARED_VERSION) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "not a shared container, or unsupported version");
        return -1;
    }

    nregions    = get64(p, 2);
    nchunks     = get64(p, 3);
    chunk_bytes = get64(p, 4);
    total_bytes = get64(p, 5);

    if (total_bytes != (uint64_t)out_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "shared container records %llu logical bytes but the read "
                "buffer is %zu (recorded shape disagrees with the writer?)",
                (unsigned long long)total_bytes, out_bytes);
        return -1;
    }
    if (nchunks == 0 || chunk_bytes == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "corrupt shared container (nchunks=%llu chunk_bytes=%llu)",
                (unsigned long long)nchunks, (unsigned long long)chunk_bytes);
        return -1;
    }

    for (uint64_t r = 0; r < nregions; r++) {
        size_t w = VOL_SHARED_HDR_WORDS + r * VOL_SHARED_DIR_WORDS;
        if ((w + VOL_SHARED_DIR_WORDS) * sizeof(uint64_t) > cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "truncated region directory in shared container");
            return -1;
        }
        uint64_t kind = get64(p, w);
        uint64_t off  = get64(p, w + 2);
        uint64_t len  = get64(p, w + 3);
        if (off + len > cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "region %llu [%llu,%llu) exceeds container size %zu",
                    (unsigned long long)r, (unsigned long long)off,
                    (unsigned long long)(off + len), cont_bytes);
            return -1;
        }
        if      (kind == VOL_REGION_SHARED_META) { meta_off = off;    meta_len = len; }
        else if (kind == VOL_REGION_CHUNK_TABLE) { table_off = off;   table_len = len; }
        else if (kind == VOL_REGION_PAYLOAD)     { payload_off = off; payload_len = len; }
        /* VOL_REGION_LAYER and anything else: ignored by this version. */
    }

    if (table_len != nchunks * sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "chunk table is %zu bytes, expected %llu",
                table_len, (unsigned long long)(nchunks * sizeof(uint64_t)));
        return -1;
    }

    /* ---- attach the shared metadata BEFORE decoding any chunk ---- */
    if (meta_len > 0) {
        prov = vol_shared_provider_for(ctx);
        if (!prov) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "container carries %zu bytes of shared metadata but no "
                    "provider claims codec '%s' in this build (providers: %s)",
                    meta_len, ctx->compressor_id, vol_shared_providers_list());
            return -1;
        }
        if (prov->attach(ctx, p + meta_off, meta_len) < 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "provider '%s' failed to attach %zu bytes of shared "
                    "metadata", prov->name, meta_len);
            return -1;
        }
    }

    /* ---- decode chunks ---- */
    {
        const unsigned char *tbl = p + table_off;
        size_t off = payload_off;

        for (uint64_t k = 0; k < nchunks && ret == 0; k++) {
            uint64_t csz;
            memcpy(&csz, tbl + k * sizeof(uint64_t), sizeof(uint64_t));

            if (csz == 0 || off + csz > payload_off + payload_len) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "chunk %llu payload overruns the payload region "
                        "(container corrupt?)", (unsigned long long)k);
                ret = -1;
                break;
            }

            size_t doff = (size_t)k * (size_t)chunk_bytes;
            size_t dlen = (out_bytes - doff < (size_t)chunk_bytes)
                              ? (out_bytes - doff) : (size_t)chunk_bytes;
            int rc;

            if (prov)
                rc = prov->decompress_chunk(ctx, p + off, (size_t)csz,
                                            (unsigned char *)out + doff, dlen);
            else
                rc = (int)H5VL_pass_through_ext_transfer_decompress_chunk(
                         ctx, p + off, (size_t)csz,
                         (unsigned char *)out + doff, dlen);

            if (rc < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "shared-container decompression failed at chunk %llu/%llu "
                        "for '%s'", (unsigned long long)k,
                        (unsigned long long)nchunks, ctx->compressor_id);
                ret = -1;
                break;
            }
            off += (size_t)csz;
        }
    }

    if (prov && prov->release) prov->release(ctx);
    return ret;
}

extern "C" herr_t
H5VL_pass_through_ext_decompress_shared(compression_ctx *ctx,
                                        const void *cbuf, size_t cont_bytes,
                                        void *out, size_t out_bytes)
{
    try {
        return vol_decompress_shared_impl(ctx, cbuf, cont_bytes, out, out_bytes);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "shared decompress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "shared decompress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}