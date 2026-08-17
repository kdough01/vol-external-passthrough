#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#if defined(__has_include)
#  if __has_include("H5VLpassthru_ext_private.h")
#    include "H5VLpassthru_ext_private.h"
#  endif
#endif
#include "metadata_structs.h"
#include "vol_errors.h"
#include "vol_shared_meta.h"
#include "vol_progressive.h"
#include "vol_container_read.h"
}

#ifndef VOL_NATIVE_MAGIC
#  error "VOL_NATIVE_MAGIC not visible -- include whichever header defines the \
container magics (grep -rn VOL_NATIVE_MAGIC src/*.h)"
#endif

extern "C" size_t vol_logical_nbytes(const compression_ctx *ctx);

namespace {

inline uint64_t get64(const unsigned char *p, size_t w)
{ uint64_t v; memcpy(&v, p + w * sizeof(uint64_t), sizeof(uint64_t)); return v; }

/* Grow-on-demand range list. Adjacent ranges are coalesced: fewer, larger
 * hyperslab reads beat many small ones on a parallel filesystem, and the
 * chunk payloads we select are usually contiguous runs anyway. */
struct range_list {
    vol_range_t *v = nullptr;
    size_t n = 0, cap = 0;

    int add(uint64_t off, uint64_t len) {
        if (len == 0) return 0;
        if (n && v[n - 1].off + v[n - 1].len == off) {   /* coalesce */
            v[n - 1].len += len;
            return 0;
        }
        if (n == cap) {
            size_t nc = cap ? cap * 2 : 8;
            vol_range_t *nv = (vol_range_t *)realloc(v, nc * sizeof(vol_range_t));
            if (!nv) return -1;
            v = nv; cap = nc;
        }
        v[n].off = off; v[n].len = len; n++;
        return 0;
    }
};

/* Underlying container size, in bytes. */
herr_t container_size(void *under, hid_t under_vol_id, hid_t plist_id,
                      uint64_t *out)
{
    H5VL_dataset_get_args_t ga;
    ga.op_type = H5VL_DATASET_GET_SPACE;
    if (H5VLdataset_get(under, under_vol_id, &ga, plist_id, NULL) < 0)
        return -1;
    hid_t fs = ga.args.get_space.space_id;
    hssize_t np = H5Sget_simple_extent_npoints(fs);
    H5Sclose(fs);
    if (np < 0) return -1;
    *out = (uint64_t)np;
    return 0;
}

/* Logical byte range this read touches, from the file selection. Mirrors the
 * contiguity rule the write path enforces; anything else falls back to the
 * whole array rather than risking a wrong answer. */
void logical_span(const compression_ctx *ctx, hid_t file_space_id,
                  uint64_t total_bytes, uint64_t *lo, uint64_t *hi)
{
    *lo = 0; *hi = total_bytes;

    if (file_space_id == H5S_ALL || !ctx || ctx->ndims == 0 || !ctx->dims)
        return;

    hssize_t np = H5Sget_select_npoints(file_space_id);
    if (np <= 0) return;

    hsize_t start[H5S_MAX_RANK], end[H5S_MAX_RANK];
    if (H5Sget_select_bounds(file_space_id, start, end) < 0) return;

    hsize_t block = 1;
    int contiguous = 1;
    for (size_t i = 0; i < ctx->ndims; i++) {
        block *= (end[i] - start[i] + 1);
        if (i >= 1 && (start[i] != 0 || end[i] != ctx->dims[i] - 1))
            contiguous = 0;
    }
    if (!contiguous || block != (hsize_t)np) return;   /* keep [0,total) */

    hsize_t lin = 0, stride = 1;
    for (int i = (int)ctx->ndims - 1; i >= 0; i--) {
        lin += start[i] * stride;
        stride *= ctx->dims[i];
    }
    size_t dsize = pressio_dtype_size(ctx->dtype);
    *lo = (uint64_t)lin * dsize;
    *hi = *lo + (uint64_t)np * dsize;
    if (*hi > total_bytes) *hi = total_bytes;
}

} /* anonymous namespace */

extern "C" {

herr_t
vol_container_read_range(void *under, hid_t under_vol_id, hid_t plist_id,
                         uint64_t cont_bytes, uint64_t off, uint64_t len,
                         void *dst)
{
    if (len == 0) return 0;
    if (off + len > cont_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "container range [%llu,%llu) exceeds container size %llu",
                (unsigned long long)off, (unsigned long long)(off + len),
                (unsigned long long)cont_bytes);
        return -1;
    }

    hsize_t mlen = (hsize_t)len;
    hsize_t clen = (hsize_t)cont_bytes;
    hsize_t coff = (hsize_t)off;

    hid_t mspace = H5Screate_simple(1, &mlen, NULL);
    hid_t fspace = H5Screate_simple(1, &clen, NULL);
    if (mspace < 0 || fspace < 0) {
        if (mspace >= 0) H5Sclose(mspace);
        if (fspace >= 0) H5Sclose(fspace);
        return -1;
    }
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &coff, NULL, &mlen, NULL);

    /* Named arrays, not compound literals: (hid_t[]){...} is C99 and is fine
     * in H5VLpassthru_ext.c, but this is C++ where taking the address of that
     * temporary is ill-formed. */
    hid_t  mtypes[1] = { H5T_NATIVE_UCHAR };
    void  *bufs[1]   = { dst };
    void  *unders[1] = { under };

    herr_t rc = H5VLdataset_read(1, unders, under_vol_id, mtypes,
                                 &mspace, &fspace, plist_id, bufs, NULL);
    H5Sclose(mspace);
    H5Sclose(fspace);

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "failed reading container range [%llu,%llu)",
                (unsigned long long)off, (unsigned long long)(off + len));
    return rc;
}

void
vol_read_plan_free(vol_read_plan_t *plan)
{
    if (!plan) return;
    free(plan->ranges);
    plan->ranges = NULL;
    plan->nranges = 0;
}

herr_t
vol_container_plan(void *under, hid_t under_vol_id, hid_t plist_id,
                   const compression_ctx *ctx,
                   hid_t file_space_id, unsigned want_pct,
                   vol_read_plan_t *plan)
{
    unsigned char  hdr[4096];
    unsigned char *dir = NULL;
    uint64_t       cont_bytes = 0;
    range_list     rl;
    herr_t         ret = 0;

    if (!plan) return -1;
    memset(plan, 0, sizeof(*plan));
    plan->want_pct = 100;

    if (container_size(under, under_vol_id, plist_id, &cont_bytes) < 0 ||
        cont_bytes < sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "could not determine container size, or it is too small");
        return -1;
    }
    plan->cont_bytes = cont_bytes;

    const uint64_t probe = (cont_bytes < sizeof(hdr)) ? cont_bytes
                                                      : (uint64_t)sizeof(hdr);
    if (vol_container_read_range(under, under_vol_id, plist_id, cont_bytes,
                                 0, probe, hdr) < 0)
        return -1;

    plan->magic = get64(hdr, 0);

    if (plan->magic == VOL_NATIVE_MAGIC) {
        plan->kind = VOL_CONT_NATIVE;

        const uint64_t hdr2  = 2 * sizeof(uint64_t);
        const uint64_t csize = (probe >= hdr2) ? get64(hdr, 1) : 0;

        /* Reduced-fidelity read. Only codecs with an embedded bitstream can
         * do this; for anyone else we silently fetch everything rather than
         * hand back a field the caller thinks is approximate but is garbage.
         *
         * NOTE: this is a contiguous front prefix, which SPERR guarantees only
         * for a SINGLE-CHUNK stream. Use VOL chunking for the multi-chunk case.
         */
        if (want_pct >= 1 && want_pct < 100 && csize > 0 && ctx &&
            vol_codec_is_progressive(ctx->compressor_id)) {

            uint64_t need = hdr2 + (csize * (uint64_t)want_pct) / 100
                                 + VOL_SPERR_SLACK;
            if (need > hdr2 + csize) need = hdr2 + csize;
            if (need > cont_bytes)   need = cont_bytes;

            plan->want_pct = want_pct;
            plan->partial  = (need < cont_bytes);
            ret = rl.add(0, need);
        } else {
            if (want_pct >= 1 && want_pct < 100 && ctx &&
                !vol_codec_is_progressive(ctx->compressor_id))
                fprintf(stderr,
                    "[VOL] progressive read requested at %u%% but compressor "
                    "'%s' has no embedded bitstream; reading at full fidelity\n",
                    want_pct, ctx->compressor_id);
            ret = rl.add(0, cont_bytes);
        }

    } else if (plan->magic == VOL_PRESSIO_MAGIC) {
        plan->kind = VOL_CONT_PRESSIO;
        if (probe >= 3 * sizeof(uint64_t))
            plan->pressio_chunk_elems = get64(hdr, 2);
        ret = rl.add(0, cont_bytes);

    } else if (plan->magic == VOL_CHUNK_MAGIC) {
        plan->kind = VOL_CONT_CHUNKED;
        if (probe < VOL_CHUNK_HDR_WORDS * sizeof(uint64_t)) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "truncated chunked header");
            return -1;
        }
        const uint64_t nchunks     = get64(hdr, 1);
        const uint64_t chunk_bytes = get64(hdr, 2);
        plan->nchunks     = nchunks;
        plan->chunk_bytes = chunk_bytes;

        if (nchunks == 0 || chunk_bytes == 0 ||
            nchunks > (cont_bytes / sizeof(uint64_t)) + 1) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "implausible chunked header (nchunks=%llu chunk_bytes=%llu)",
                    (unsigned long long)nchunks,
                    (unsigned long long)chunk_bytes);
            return -1;
        }

        const uint64_t table_off   = VOL_CHUNK_HDR_WORDS * sizeof(uint64_t);
        const uint64_t table_bytes = nchunks * sizeof(uint64_t);
        const uint64_t payload_off = table_off + table_bytes;

        if (rl.add(0, payload_off) < 0) { ret = -1; goto done; }

        dir = (unsigned char *)malloc(table_bytes);
        if (!dir) { ret = -1; goto done; }
        if (vol_container_read_range(under, under_vol_id, plist_id, cont_bytes,
                                     table_off, table_bytes, dir) < 0) {
            ret = -1; goto done;
        }

        {
            const uint64_t total_logical = (uint64_t)vol_logical_nbytes(ctx);
            uint64_t lo, hi;
            logical_span(ctx, file_space_id, total_logical, &lo, &hi);

            const uint64_t k0 = lo / chunk_bytes;
            uint64_t k1 = (hi + chunk_bytes - 1) / chunk_bytes;
            if (k1 > nchunks) k1 = nchunks;

            /* PROGRESSIVE. Each VOL chunk is its own codec stream, so a prefix
             * of one chunk is exactly what SPERR's progressive_truncate wants --
             * the single-chunk case NCAR documents as safe to cut anywhere.
             * vol_read_vol recomputes this same formula, so nothing extra needs
             * to be plumbed through the plan. */
            const int prog = (ctx && vol_codec_is_progressive(ctx->compressor_id));
            if (prog) plan->want_pct = want_pct;

            uint64_t off = payload_off;
            for (uint64_t k = 0; k < nchunks; k++) {
                uint64_t csz;
                memcpy(&csz, dir + k * sizeof(uint64_t), sizeof(uint64_t));
                if (k >= k0 && k < k1) {
                    uint64_t take = csz;
                    unsigned p    = vol_progressive_pct_for_chunk(want_pct, k);
                    if (prog && p >= 1 && p < 100) {
                        take = (csz * (uint64_t)p) / 100 + VOL_SPERR_SLACK;
                        if (take > csz) take = csz;
                        plan->partial = 1;
                    }
                    if (rl.add(off, take) < 0) { ret = -1; goto done; }
                }
                off += csz;
            }
            /* OR, not assign. The original `plan->partial = (k0 != 0 || ...)`
             * would clobber the flag the loop just set for fidelity truncation:
             * a container partial ONLY because of truncation would come back
             * marked partial=0, vol_container_fetch would skip zeroing the
             * gaps, and stale heap bytes would reach the decoder. */
            if (k0 != 0 || k1 != nchunks) plan->partial = 1;
        }

    } else if (plan->magic == VOL_SHARED_MAGIC) {
        plan->kind = VOL_CONT_SHARED;
        const uint64_t nregions = get64(hdr, 2);
        plan->nchunks     = get64(hdr, 3);
        plan->chunk_bytes = get64(hdr, 4);

        const uint64_t dir_bytes =
            nregions * VOL_SHARED_DIR_WORDS * sizeof(uint64_t);
        const uint64_t dir_end = VOL_SHARED_HDR_WORDS * sizeof(uint64_t) + dir_bytes;
        if (dir_end > cont_bytes || nregions == 0 || nregions > 1024) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "implausible shared-container directory (%llu regions)",
                    (unsigned long long)nregions);
            return -1;
        }
        if (dir_end > probe) {
            if (vol_container_read_range(under, under_vol_id, plist_id,
                                         cont_bytes, probe, dir_end - probe,
                                         hdr + probe) < 0)
                return -1;
        }
        if (rl.add(0, dir_end) < 0) { ret = -1; goto done; }

        uint64_t payload_off = 0, payload_len = 0, table_off = 0;
        for (uint64_t r = 0; r < nregions; r++) {
            size_t w = VOL_SHARED_HDR_WORDS + r * VOL_SHARED_DIR_WORDS;
            uint64_t kind = get64(hdr, w);
            uint64_t off  = get64(hdr, w + 2);
            uint64_t len  = get64(hdr, w + 3);
            if (off + len > cont_bytes) { ret = -1; goto done; }
            if (kind == VOL_REGION_SHARED_META) {
                if (rl.add(off, len) < 0) { ret = -1; goto done; }
            } else if (kind == VOL_REGION_CHUNK_TABLE) {
                table_off = off;
                if (rl.add(off, len) < 0) { ret = -1; goto done; }
            } else if (kind == VOL_REGION_PAYLOAD) {
                payload_off = off; payload_len = len;
            }
        }

        if (plan->nchunks && plan->chunk_bytes && table_off) {
            const uint64_t table_bytes = plan->nchunks * sizeof(uint64_t);
            dir = (unsigned char *)malloc(table_bytes);
            if (!dir) { ret = -1; goto done; }
            if (vol_container_read_range(under, under_vol_id, plist_id,
                                         cont_bytes, table_off, table_bytes,
                                         dir) < 0) { ret = -1; goto done; }

            const uint64_t total_logical = (uint64_t)vol_logical_nbytes(ctx);
            uint64_t lo, hi;
            logical_span(ctx, file_space_id, total_logical, &lo, &hi);

            const uint64_t k0 = lo / plan->chunk_bytes;
            uint64_t k1 = (hi + plan->chunk_bytes - 1) / plan->chunk_bytes;
            if (k1 > plan->nchunks) k1 = plan->nchunks;

            uint64_t off = payload_off;
            for (uint64_t k = 0; k < plan->nchunks; k++) {
                uint64_t csz;
                memcpy(&csz, dir + k * sizeof(uint64_t), sizeof(uint64_t));
                if (k >= k0 && k < k1) {
                    if (rl.add(off, csz) < 0) { ret = -1; goto done; }
                }
                off += csz;
            }
            plan->partial = (k0 != 0 || k1 != plan->nchunks);
        } else {
            if (rl.add(payload_off, payload_len) < 0) { ret = -1; goto done; }
        }

    } else {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "unrecognized container magic 0x%016llx "
                "(file written by an incompatible VOL version?)",
                (unsigned long long)plan->magic);
        return -1;
    }

done:
    free(dir);
    if (ret < 0) { free(rl.v); return -1; }

    plan->ranges  = rl.v;
    plan->nranges = rl.n;
    plan->bytes_needed = 0;
    for (size_t i = 0; i < rl.n; i++) plan->bytes_needed += rl.v[i].len;

    {
        const char *pcsv = getenv("VOL_PLAN_CSV");
        if (pcsv && *pcsv) {
            FILE *pf = fopen(pcsv, "a");
            if (pf) {
                /* kind,want_pct,nranges,bytes_needed,cont_bytes,partial */
                fprintf(pf, "%d,%u,%zu,%llu,%llu,%d\n",
                        (int)plan->kind, plan->want_pct, plan->nranges,
                        (unsigned long long)plan->bytes_needed,
                        (unsigned long long)plan->cont_bytes, plan->partial);
                fclose(pf);
            }
        }
    }

    if (getenv("VOL_READ_PLAN_LOG"))
        fprintf(stderr,
            "[read-plan] kind=%d pct=%u ranges=%zu need=%llu/%llu bytes "
            "(%.1f%%)%s\n",
            (int)plan->kind, plan->want_pct, plan->nranges,
            (unsigned long long)plan->bytes_needed,
            (unsigned long long)plan->cont_bytes,
            100.0 * (double)plan->bytes_needed / (double)plan->cont_bytes,
            plan->partial ? "  PARTIAL" : "");
    return 0;
}

herr_t vol_container_fetch(void *under, hid_t under_vol_id, hid_t plist_id,
                           compression_ctx *ctx,
                           const vol_read_plan_t *plan,
                           unsigned char **out_cbuf)
{
    if (!plan || !out_cbuf || !ctx) return -1;
    *out_cbuf = NULL;

    unsigned char *buf = (unsigned char *)
        H5VL_pass_through_ext_reserve_container(ctx, (size_t)plan->cont_bytes);
    if (!buf) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory reserving a %llu byte container buffer",
                (unsigned long long)plan->cont_bytes);
        return -1;
    }

    if (plan->partial) {
        vol_range_t *s = (vol_range_t *)malloc(plan->nranges * sizeof(*s));
        if (!s) {
            memset(buf, 0, (size_t)plan->cont_bytes);
        } else {
            memcpy(s, plan->ranges, plan->nranges * sizeof(*s));
            for (size_t i = 1; i < plan->nranges; i++)
                for (size_t j = i; j && s[j - 1].off > s[j].off; j--) {
                    vol_range_t tmp = s[j]; s[j] = s[j - 1]; s[j - 1] = tmp;
                }
            uint64_t cur = 0;
            for (size_t i = 0; i < plan->nranges; i++) {
                if (s[i].off > cur)
                    memset(buf + cur, 0, (size_t)(s[i].off - cur));
                if (s[i].off + s[i].len > cur)
                    cur = s[i].off + s[i].len;
            }
            if (cur < plan->cont_bytes)
                memset(buf + cur, 0, (size_t)(plan->cont_bytes - cur));
            free(s);
        }
    }

    for (size_t i = 0; i < plan->nranges; i++) {
        if (vol_container_read_range(under, under_vol_id, plist_id,
                                     plan->cont_bytes,
                                     plan->ranges[i].off, plan->ranges[i].len,
                                     buf + plan->ranges[i].off) < 0) {
            return -1;      /* buf is pooled: do NOT free */
        }
    }

    *out_cbuf = buf;
    return 0;
}

} /* extern "C" */