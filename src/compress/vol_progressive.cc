#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "hdf5.h"
#include "vol_errors.h"
#include "vol_progressive.h"
}

#include <SPERR_C_API.h>

#define VOL_PROG_PCT_NAME "vol:progressive_pct"

extern "C" {

int
vol_codec_is_progressive(const char *compressor_id)
{
    return compressor_id && strcmp(compressor_id, "sperr") == 0;
}

herr_t
H5Pset_vol_progressive_pct(hid_t dxpl_id, unsigned pct)
{
    htri_t ex;

    if (dxpl_id <= 0 || dxpl_id == H5P_DEFAULT) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_invalid_option,
                "H5Pset_vol_progressive_pct needs a real DXPL, not H5P_DEFAULT");
        return -1;
    }
    if (pct < 1)   pct = 1;
    if (pct > 100) pct = 100;

    ex = H5Pexist(dxpl_id, VOL_PROG_PCT_NAME);
    if (ex > 0)
        return H5Pset(dxpl_id, VOL_PROG_PCT_NAME, &pct);

    return H5Pinsert2(dxpl_id, VOL_PROG_PCT_NAME, sizeof(unsigned), &pct,
                      NULL, NULL, NULL, NULL, NULL, NULL);
}

unsigned
H5VL_pass_through_ext_progressive_pct(hid_t dxpl_id)
{
    unsigned pct = 100;

    if (dxpl_id > 0 && dxpl_id != H5P_DEFAULT &&
        H5Pexist(dxpl_id, VOL_PROG_PCT_NAME) > 0 &&
        H5Pget(dxpl_id, VOL_PROG_PCT_NAME, &pct) >= 0 &&
        pct >= 1 && pct <= 100)
        return pct;

    /* Env override, for benchmark harnesses that cannot set a DXPL. */
    {
        const char *e = getenv("VOL_PROGRESSIVE_PCT");
        if (e && *e) {
            long v = strtol(e, NULL, 10);
            if (v >= 1 && v <= 100) return (unsigned)v;
        }
    }
    return 100;
}

static unsigned *g_pct_vec = NULL;
static size_t    g_pct_n   = 0;

unsigned
vol_progressive_pct_for_chunk(unsigned want_pct, uint64_t k)
{
    if (g_pct_vec && k < (uint64_t)g_pct_n) {
        unsigned p = g_pct_vec[k];
        if (p >= 1 && p <= 100) return p;
    }
    return want_pct;
}

herr_t
H5Pset_vol_progressive_pct_v(hid_t dxpl_id, size_t n, const unsigned *pct)
{
    unsigned *v;

    (void)dxpl_id;
    if (n == 0 || !pct) {
        free(g_pct_vec);
        g_pct_vec = NULL;
        g_pct_n   = 0;
        return 0;
    }
    v = (unsigned *)malloc(n * sizeof(unsigned));
    if (!v) return -1;
    for (size_t i = 0; i < n; i++) {
        unsigned p = pct[i];
        if (p < 1)   p = 1;
        if (p > 100) p = 100;
        v[i] = p;
    }
    free(g_pct_vec);
    g_pct_vec = v;
    g_pct_n   = n;
    return 0;
}

herr_t
vol_sperr_truncate(const void *stream, size_t stream_len, unsigned pct,
                   void **out, size_t *out_len)
{
    void  *dst = NULL;
    size_t dst_len = 0;
    int    rc;

    if (!stream || stream_len == 0 || !out || !out_len) return -1;
    *out = NULL;
    *out_len = 0;

    rc = C_API::sperr_trunc_3d(stream, stream_len, pct, &dst, &dst_len);
    if (rc != 0 || !dst || dst_len == 0) {
        free(dst);
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "sperr_trunc_3d failed (rc=%d) at %u%% from %zu available bytes",
                rc, pct, stream_len);
        return -1;
    }

    if (getenv("VOL_PROGRESSIVE_LOG"))
        fprintf(stderr, "[prog] sperr_trunc_3d: %zu B in, %u%% -> %zu B out\n",
                stream_len, pct, dst_len);

    *out     = dst;
    *out_len = dst_len;
    return 0;
}

} /* extern "C" */