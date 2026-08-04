#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <exception>

extern "C" {
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"
#include "vol_errors.h"
#include "vol_shared_meta.h"
#include "vol_progressive.h"
}

/* Reused per layer: each layer is a whole-dataset compress/decompress at a
 * chosen absolute bound, which is exactly what the native path already does. */
extern "C" herr_t H5VL_pass_through_ext_compress_native(
    compression_ctx *ctx, const void *data, size_t nbytes, size_t hdr_reserve,
    void **out_cbuf, uint64_t *out_csize);
extern "C" herr_t H5VL_pass_through_ext_decompress_native(
    compression_ctx *ctx, const void *cbuf, size_t csize,
    void *out, size_t out_bytes);
extern "C" size_t vol_logical_nbytes(const compression_ctx *ctx);

namespace {

inline void put64(unsigned char *p, size_t w, uint64_t v)
{ memcpy(p + w * sizeof(uint64_t), &v, sizeof(uint64_t)); }
inline uint64_t get64(const unsigned char *p, size_t w)
{ uint64_t v; memcpy(&v, p + w * sizeof(uint64_t), sizeof(uint64_t)); return v; }

/* ---- typed elementwise helpers -------------------------------------- */

void vol_sub(void *dst, const void *a, const void *b, size_t nelem,
             enum pressio_dtype dt)
{
    if (dt == pressio_double_dtype) {
        double *d = (double *)dst; const double *x = (const double *)a;
        const double *y = (const double *)b;
        for (size_t i = 0; i < nelem; i++) d[i] = x[i] - y[i];
    } else {
        float *d = (float *)dst; const float *x = (const float *)a;
        const float *y = (const float *)b;
        for (size_t i = 0; i < nelem; i++) d[i] = x[i] - y[i];
    }
}

void vol_add_inplace(void *acc, const void *x, size_t nelem,
                     enum pressio_dtype dt)
{
    if (dt == pressio_double_dtype) {
        double *a = (double *)acc; const double *b = (const double *)x;
        for (size_t i = 0; i < nelem; i++) a[i] += b[i];
    } else {
        float *a = (float *)acc; const float *b = (const float *)x;
        for (size_t i = 0; i < nelem; i++) a[i] += b[i];
    }
}

void vol_minmax(const void *data, size_t nelem, enum pressio_dtype dt,
                double *mn, double *mx)
{
    double lo = DBL_MAX, hi = -DBL_MAX;
    if (dt == pressio_double_dtype) {
        const double *p = (const double *)data;
        for (size_t i = 0; i < nelem; i++) {
            if (p[i] < lo) lo = p[i];
            if (p[i] > hi) hi = p[i];
        }
    } else {
        const float *p = (const float *)data;
        for (size_t i = 0; i < nelem; i++) {
            double v = (double)p[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }
    *mn = lo; *mx = hi;
}

/* Set the absolute error bound for the next compress call. pressio:abs is the
 * generic key every error-bounded plugin in this build honours, which is what
 * keeps the scheme codec-agnostic. */
int vol_set_abs(compression_ctx *ctx, double abs_bound)
{
    struct pressio_options *o = pressio_options_new();
    if (!o) return -1;
    pressio_options_set_double(o, "pressio:abs", abs_bound);
    int rc = pressio_compressor_set_options(ctx->compressor, o);
    pressio_options_free(o);
    return rc;
}

/* Read the configured target bound. Prefers pressio:abs; falls back to
 * pressio:rel scaled by the global range. */
int vol_target_bound(compression_ctx *ctx, double range, double *out)
{
    struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
    double v = 0.0;
    int found = 0;

    if (o) {
        if (pressio_options_get_double(o, "pressio:abs", &v) ==
                pressio_options_key_set && v > 0.0) {
            *out = v; found = 1;
        } else if (pressio_options_get_double(o, "pressio:rel", &v) ==
                       pressio_options_key_set && v > 0.0) {
            *out = v * range; found = 1;
        }
        pressio_options_free(o);
    }
    return found ? 0 : -1;
}

} /* anonymous namespace */

extern "C" {

int
H5VL_pass_through_ext_progressive_layers(const compression_ctx *ctx)
{
    const char *e = getenv("VOL_PROGRESSIVE_LAYERS");
    long v = 0;
    if (e && *e) v = strtol(e, NULL, 10);
    if (v <= 0 && ctx) {
        /* stashed by parse_chunking_opts from vol:progressive_layers */
        v = (long)ctx->chunk_n;      /* reused field; see integration notes */
    }
    if (v <= 0) v = VOL_PROGRESSIVE_DEFAULT_LAYERS;
    if (v > VOL_PROGRESSIVE_MAX_LAYERS) v = VOL_PROGRESSIVE_MAX_LAYERS;
    return (int)v;
}

double
H5VL_pass_through_ext_progressive_ratio(const compression_ctx *ctx)
{
    (void)ctx;
    const char *e = getenv("VOL_PROGRESSIVE_RATIO");
    double v = 0.0;
    if (e && *e) v = strtod(e, NULL);
    if (v <= 1.0) v = VOL_PROGRESSIVE_DEFAULT_RATIO;
    return v;
}

int
H5VL_pass_through_ext_progressive_want(hid_t dxpl_id)
{
    int want = 0;

    if (dxpl_id > 0 && dxpl_id != H5P_DEFAULT) {
        htri_t ex = H5Pexist(dxpl_id, "vol:progressive_layers");
        if (ex > 0 && H5Pget(dxpl_id, "vol:progressive_layers", &want) >= 0 &&
            want > 0)
            return want;
    }

    const char *e = getenv("VOL_PROGRESSIVE_LAYERS_READ");
    if (e && *e) {
        long v = strtol(e, NULL, 10);
        if (v > 0) return (int)v;
    }
    return 0;                 /* 0 => every layer, i.e. full fidelity */
}

/* =========================================================================
 * WRITE
 * ========================================================================= */
static herr_t
vol_compress_progressive_impl(compression_ctx *ctx,
                              const void *data, size_t nbytes,
                              int nlayers, double ratio,
                              void **out_container, uint64_t *out_total)
{
    herr_t    ret     = 0;
    void     *acc     = NULL;    /* running reconstruction              */
    void     *work    = NULL;    /* residual, then decoded residual     */
    void    **layers  = NULL;
    uint64_t *lsizes  = NULL;
    double   *bounds  = NULL;
    double    saved_abs = 0.0;
    int       have_saved = 0;
    size_t    elem_size, nelem;
    double    gmin = 0.0, gmax = 0.0, target = 0.0;

    if (!ctx || !ctx->compressor || !data || !out_container || !out_total ||
        nbytes == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "invalid arguments to compress_progressive");
        return -1;
    }
    *out_container = NULL;
    *out_total     = 0;

    if (ctx->dtype != pressio_float_dtype && ctx->dtype != pressio_double_dtype) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "progressive mode requires float or double data; residual "
                "layers are not meaningful for integer or byte data");
        return -1;
    }
    if (vol_logical_nbytes(ctx) != nbytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "shape/size mismatch: recorded dims imply %zu bytes, buffer is %zu",
                vol_logical_nbytes(ctx), nbytes);
        return -1;
    }

    elem_size = pressio_dtype_size(ctx->dtype);
    nelem     = nbytes / elem_size;
    if (nlayers < 1) nlayers = 1;
    if (nlayers > VOL_PROGRESSIVE_MAX_LAYERS) nlayers = VOL_PROGRESSIVE_MAX_LAYERS;
    if (ratio <= 1.0) ratio = VOL_PROGRESSIVE_DEFAULT_RATIO;

    /* Global statistics: needed to turn a relative target into an absolute
     * one, and stored so the reader can report achieved accuracy. A filter
     * chunk could not compute this -- it only ever sees its own bytes. */
    vol_minmax(data, nelem, ctx->dtype, &gmin, &gmax);
    if (vol_target_bound(ctx, gmax - gmin, &target) < 0 || target <= 0.0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "progressive mode needs an absolute or relative error bound; "
                "neither pressio:abs nor pressio:rel is set on '%s'",
                ctx->compressor_id);
        return -1;
    }

    /* Snapshot the caller's bound so we can restore it afterwards. */
    {
        struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
        if (o) {
            if (pressio_options_get_double(o, "pressio:abs", &saved_abs) ==
                    pressio_options_key_set)
                have_saved = 1;
            pressio_options_free(o);
        }
    }

    bounds = (double *)malloc((size_t)nlayers * sizeof(double));
    layers = (void **)calloc((size_t)nlayers, sizeof(void *));
    lsizes = (uint64_t *)calloc((size_t)nlayers, sizeof(uint64_t));
    acc    = calloc(1, nbytes);          /* recon starts at zero */
    work   = malloc(nbytes);

    if (!bounds || !layers || !lsizes || !acc || !work) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "out of memory for %d progressive layers over %zu bytes "
                "(needs ~2x the logical size in working buffers)",
                nlayers, nbytes);
        ret = -1;
        goto done;
    }

    /* Coarse -> fine, geometric, ending exactly at the configured target. */
    for (int k = 0; k < nlayers; k++)
        bounds[k] = target * pow(ratio, (double)(nlayers - 1 - k));

    if (getenv("VOL_PROGRESSIVE_LOG")) {
        fprintf(stderr, "[prog] '%s' %d layers, ratio %.1f, target %.6e, "
                        "range [%.6e, %.6e]\n  bounds:",
                ctx->compressor_id, nlayers, ratio, target, gmin, gmax);
        for (int k = 0; k < nlayers; k++) fprintf(stderr, " %.3e", bounds[k]);
        fprintf(stderr, "\n");
    }

    ctx->device_ms       = 0.0;
    ctx->pressio_call_ms = 0.0;

    for (int k = 0; k < nlayers && ret == 0; k++) {
        /* residual = original - reconstruction so far */
        vol_sub(work, data, acc, nelem, ctx->dtype);

        if (vol_set_abs(ctx, bounds[k]) != 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "could not set pressio:abs=%.6e on '%s' for layer %d: %s",
                    bounds[k], ctx->compressor_id, k,
                    pressio_compressor_error_msg(ctx->compressor));
            ret = -1;
            break;
        }

        if (H5VL_pass_through_ext_compress_native(ctx, work, nbytes, 0,
                                                  &layers[k], &lsizes[k]) < 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "progressive layer %d/%d failed to compress", k, nlayers);
            ret = -1;
            break;
        }

        /* Decode it back so the next residual is against what a READER will
         * actually reconstruct -- not against the exact residual, which would
         * accumulate the quantisation error we just introduced. */
        if (k + 1 < nlayers) {
            if (H5VL_pass_through_ext_decompress_native(ctx, layers[k],
                                                        (size_t)lsizes[k],
                                                        work, nbytes) < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compress_failed,
                        "progressive layer %d/%d failed to round-trip", k, nlayers);
                ret = -1;
                break;
            }
            vol_add_inplace(acc, work, nelem, ctx->dtype);
        }

        if (getenv("VOL_PROGRESSIVE_LOG"))
            fprintf(stderr, "[prog]   layer %d bound=%.3e -> %llu B (%.2fx)\n",
                    k, bounds[k], (unsigned long long)lsizes[k],
                    (double)nbytes / (double)lsizes[k]);
    }

    if (ret == 0) {
        const uint64_t nregions = (uint64_t)nlayers + 1;
        const size_t hdr_bytes =
            (VOL_PROGRESSIVE_HDR_WORDS + nregions * VOL_SHARED_DIR_WORDS)
            * sizeof(uint64_t);
        const size_t meta_bytes = ((size_t)nlayers + 2) * sizeof(double);
        size_t payload_bytes = 0;
        for (int k = 0; k < nlayers; k++) payload_bytes += (size_t)lsizes[k];

        const size_t meta_off = hdr_bytes;
        const size_t total    = meta_off + meta_bytes + payload_bytes;

        unsigned char *c = (unsigned char *)malloc(total);
        if (!c) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory assembling a %zu byte progressive container",
                    total);
            ret = -1;
        } else {
            put64(c, 0, VOL_PROGRESSIVE_MAGIC);
            put64(c, 1, VOL_PROGRESSIVE_VERSION);
            put64(c, 2, nregions);
            put64(c, 3, (uint64_t)nlayers);
            put64(c, 4, (uint64_t)elem_size);
            put64(c, 5, (uint64_t)nbytes);
            put64(c, 6, 0);
            put64(c, 7, 0);

            size_t w = VOL_PROGRESSIVE_HDR_WORDS;
            put64(c, w + 0, VOL_REGION_SHARED_META);
            put64(c, w + 1, 0);
            put64(c, w + 2, (uint64_t)meta_off);
            put64(c, w + 3, (uint64_t)meta_bytes);
            w += VOL_SHARED_DIR_WORDS;

            size_t off = meta_off + meta_bytes;
            for (int k = 0; k < nlayers; k++) {
                put64(c, w + 0, VOL_REGION_LAYER);
                put64(c, w + 1, (uint64_t)k);
                put64(c, w + 2, (uint64_t)off);
                put64(c, w + 3, lsizes[k]);
                w += VOL_SHARED_DIR_WORDS;
                off += (size_t)lsizes[k];
            }

            /* meta: bounds[nlayers], gmin, gmax */
            memcpy(c + meta_off, bounds, (size_t)nlayers * sizeof(double));
            memcpy(c + meta_off + (size_t)nlayers * sizeof(double),
                   &gmin, sizeof(double));
            memcpy(c + meta_off + ((size_t)nlayers + 1) * sizeof(double),
                   &gmax, sizeof(double));

            off = meta_off + meta_bytes;
            for (int k = 0; k < nlayers; k++) {
                memcpy(c + off, layers[k], (size_t)lsizes[k]);
                off += (size_t)lsizes[k];
            }

            *out_container = c;
            *out_total     = (uint64_t)total;

            if (getenv("VOL_PROGRESSIVE_LOG"))
                fprintf(stderr,
                    "[prog] container %zu B for %zu B logical (%.3fx overall; "
                    "layer 0 alone is %.3fx)\n",
                    total, nbytes, (double)nbytes / (double)total,
                    (double)nbytes / (double)lsizes[0]);
        }
    }

done:
    if (have_saved) (void)vol_set_abs(ctx, saved_abs);
    if (layers) for (int k = 0; k < nlayers; k++) free(layers[k]);
    free(layers); free(lsizes); free(bounds);
    free(acc); free(work);
    return ret;
}

herr_t
H5VL_pass_through_ext_compress_progressive(compression_ctx *ctx,
                                           const void *data, size_t nbytes,
                                           int nlayers, double ratio,
                                           void **out_container,
                                           uint64_t *out_total)
{
    try {
        return vol_compress_progressive_impl(ctx, data, nbytes, nlayers, ratio,
                                             out_container, out_total);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "progressive compress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "progressive compress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

/* =========================================================================
 * PROBE / READ
 * ========================================================================= */
herr_t
H5VL_pass_through_ext_progressive_probe(const void *cbuf, size_t cont_bytes,
                                        uint64_t *out_nlayers,
                                        uint64_t *out_total_bytes)
{
    const unsigned char *p = (const unsigned char *)cbuf;
    if (!cbuf || cont_bytes < VOL_PROGRESSIVE_HDR_WORDS * sizeof(uint64_t))
        return -1;
    if (get64(p, 0) != VOL_PROGRESSIVE_MAGIC)   return -1;
    if (get64(p, 1) != VOL_PROGRESSIVE_VERSION) return -1;
    if (out_nlayers)     *out_nlayers     = get64(p, 3);
    if (out_total_bytes) *out_total_bytes = get64(p, 5);
    return 0;
}

static herr_t
vol_decompress_progressive_impl(compression_ctx *ctx,
                                const void *cbuf, size_t cont_bytes,
                                void *out, size_t out_bytes,
                                int want_layers, double *out_achieved_bound)
{
    const unsigned char *p = (const unsigned char *)cbuf;
    herr_t   ret   = 0;
    void    *work  = NULL;
    uint64_t nregions, nlayers, elem_size, total_bytes;
    size_t   meta_off = 0, meta_len = 0, nelem;
    size_t   loff[VOL_PROGRESSIVE_MAX_LAYERS];
    size_t   llen[VOL_PROGRESSIVE_MAX_LAYERS];
    int      have[VOL_PROGRESSIVE_MAX_LAYERS];
    double   saved_abs = 0.0;
    int      have_saved = 0;

    memset(have, 0, sizeof(have));

    if (!ctx || !cbuf || !out ||
        cont_bytes < VOL_PROGRESSIVE_HDR_WORDS * sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "invalid arguments to decompress_progressive");
        return -1;
    }
    if (get64(p, 0) != VOL_PROGRESSIVE_MAGIC ||
        get64(p, 1) != VOL_PROGRESSIVE_VERSION) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "not a progressive container, or unsupported version");
        return -1;
    }

    nregions    = get64(p, 2);
    nlayers     = get64(p, 3);
    elem_size   = get64(p, 4);
    total_bytes = get64(p, 5);

    if (nlayers == 0 || nlayers > VOL_PROGRESSIVE_MAX_LAYERS) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "progressive container reports %llu layers (max %d)",
                (unsigned long long)nlayers, VOL_PROGRESSIVE_MAX_LAYERS);
        return -1;
    }
    if (total_bytes != (uint64_t)out_bytes ||
        elem_size != (uint64_t)pressio_dtype_size(ctx->dtype)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "progressive container records %llu bytes of %llu-byte elements "
                "but the read buffer is %zu bytes of %zu-byte elements",
                (unsigned long long)total_bytes, (unsigned long long)elem_size,
                out_bytes, pressio_dtype_size(ctx->dtype));
        return -1;
    }

    for (uint64_t r = 0; r < nregions; r++) {
        size_t w = VOL_PROGRESSIVE_HDR_WORDS + r * VOL_SHARED_DIR_WORDS;
        if ((w + VOL_SHARED_DIR_WORDS) * sizeof(uint64_t) > cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "truncated region directory in progressive container");
            return -1;
        }
        uint64_t kind  = get64(p, w);
        uint64_t flags = get64(p, w + 1);
        uint64_t off   = get64(p, w + 2);
        uint64_t len   = get64(p, w + 3);

        if (off + len > cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "region %llu overruns the progressive container",
                    (unsigned long long)r);
            return -1;
        }
        if (kind == VOL_REGION_SHARED_META) { meta_off = off; meta_len = len; }
        else if (kind == VOL_REGION_LAYER && flags < nlayers) {
            loff[flags] = (size_t)off;
            llen[flags] = (size_t)len;
            have[flags] = 1;
        }
    }

    /* How many layers should we actually apply? */
    {
        int n = (want_layers > 0 && (uint64_t)want_layers < nlayers)
                    ? want_layers : (int)nlayers;
        for (int k = 0; k < n; k++) {
            if (!have[k] || llen[k] == 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "progressive container is missing layer %d of %llu",
                        k, (unsigned long long)nlayers);
                return -1;
            }
        }
        want_layers = n;
    }

    if (out_achieved_bound) {
        *out_achieved_bound = 0.0;
        if (meta_len >= (size_t)(want_layers) * sizeof(double)) {
            double b;
            memcpy(&b, p + meta_off + (size_t)(want_layers - 1) * sizeof(double),
                   sizeof(double));
            *out_achieved_bound = b;
        }
    }

    nelem = out_bytes / (size_t)elem_size;
    work  = malloc(out_bytes);
    if (!work) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating a %zu byte progressive work buffer",
                out_bytes);
        return -1;
    }

    /* Each layer was encoded at its own bound; the decoder must be configured
     * the same way, so restore the per-layer bound before decoding it. */
    {
        struct pressio_options *o = pressio_compressor_get_options(ctx->compressor);
        if (o) {
            if (pressio_options_get_double(o, "pressio:abs", &saved_abs) ==
                    pressio_options_key_set)
                have_saved = 1;
            pressio_options_free(o);
        }
    }

    memset(out, 0, out_bytes);

    for (int k = 0; k < want_layers && ret == 0; k++) {
        if (meta_len >= (size_t)(k + 1) * sizeof(double)) {
            double b;
            memcpy(&b, p + meta_off + (size_t)k * sizeof(double), sizeof(double));
            (void)vol_set_abs(ctx, b);
        }

        if (H5VL_pass_through_ext_decompress_native(ctx, p + loff[k], llen[k],
                                                    work, out_bytes) < 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "progressive layer %d/%d failed to decompress",
                    k, want_layers);
            ret = -1;
            break;
        }
        vol_add_inplace(out, work, nelem, ctx->dtype);
    }

    if (getenv("VOL_PROGRESSIVE_LOG"))
        fprintf(stderr, "[prog] read %d of %llu layers%s\n",
                want_layers, (unsigned long long)nlayers,
                ((uint64_t)want_layers < nlayers) ? "  (REDUCED FIDELITY)" : "");

    if (have_saved) (void)vol_set_abs(ctx, saved_abs);
    free(work);
    return ret;
}

herr_t
H5VL_pass_through_ext_decompress_progressive(compression_ctx *ctx,
                                             const void *cbuf, size_t cont_bytes,
                                             void *out, size_t out_bytes,
                                             int want_layers,
                                             double *out_achieved_bound)
{
    try {
        return vol_decompress_progressive_impl(ctx, cbuf, cont_bytes, out,
                                               out_bytes, want_layers,
                                               out_achieved_bound);
    } catch (const std::exception &e) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "progressive decompress for '%s' threw: %s",
                ctx ? ctx->compressor_id : "?", e.what());
        return -1;
    } catch (...) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "progressive decompress for '%s' threw a non-standard exception",
                ctx ? ctx->compressor_id : "?");
        return -1;
    }
}

} /* extern "C" */