#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <libpressio/libpressio.h>
#include "hdf5.h"
#include "H5VLpassthru_ext.h"
#include "metadata_structs.h"
#include "vol_errors.h"
#include "vol_shared_meta.h"
#include "vol_progressive.h"
#include "vol_container_read.h"
}

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

    void *bufs[] = { dst };
    herr_t rc = H5VLdataset_read(1, &under, under_vol_id,
                                 (hid_t[]){H5T_NATIVE_UCHAR},
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
                   hid_t file_space_id, int want_layers,
                   vol_read_plan_t *plan)
{
    unsigned char  hdr[4096];
    unsigned char *dir = NULL;      /* directory / csize table, if larger */
    uint64_t       cont_bytes = 0;
    range_list     rl;
    herr_t         ret = 0;

    if (!plan) return -1;
    memset(plan, 0, sizeof(*plan));

    if (container_size(under, under_vol_id, plist_id, &cont_bytes) < 0 ||
        cont_bytes < sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "could not determine container size, or it is too small");
        return -1;
    }
    plan->cont_bytes = cont_bytes;

    /* ---- PHASE 1: a small prefix covers every fixed header we define ---- */
    const uint64_t probe = (cont_bytes < sizeof(hdr)) ? cont_bytes
                                                      : (uint64_t)sizeof(hdr);
    if (vol_container_read_range(under, under_vol_id, plist_id, cont_bytes,
                                 0, probe, hdr) < 0)
        return -1;

    plan->magic = get64(hdr, 0);

    /* ---- dispatch ---- */
    if (plan->magic == VOL_NATIVE_MAGIC) {
        /* One sequential libpressio stream: no independently decodable
         * sub-unit exists, so the whole payload is required. */
        plan->kind = VOL_CONT_NATIVE;
        ret = rl.add(0, cont_bytes);

    } else if (plan->magic == VOL_PRESSIO_MAGIC) {
        /* Chunked inside libpressio's own format; decompress() wants it all. */
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

        /* header + csize table are always needed */
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

            uint64_t off = payload_off;
            for (uint64_t k = 0; k < nchunks; k++) {
                uint64_t csz;
                memcpy(&csz, dir + k * sizeof(uint64_t), sizeof(uint64_t));
                if (k >= k0 && k < k1) {
                    if (rl.add(off, csz) < 0) { ret = -1; goto done; }
                }
                off += csz;
            }
            plan->partial = (k0 != 0 || k1 != nchunks);
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
                return -1;      /* dir_end <= 4096 given nregions <= 1024 */
        }
        if (rl.add(0, dir_end) < 0) { ret = -1; goto done; }

        /* Shared metadata + chunk table are ALWAYS needed: every chunk
         * references the shared region, and the table gives the offsets. */
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

    } else if (plan->magic == VOL_PROGRESSIVE_MAGIC) {
        plan->kind = VOL_CONT_PROGRESSIVE;
        const uint64_t nregions = get64(hdr, 2);
        const uint64_t nlayers  = get64(hdr, 3);
        plan->nlayers = nlayers;

        const uint64_t dir_bytes =
            nregions * VOL_SHARED_DIR_WORDS * sizeof(uint64_t);
        const uint64_t dir_end =
            VOL_PROGRESSIVE_HDR_WORDS * sizeof(uint64_t) + dir_bytes;
        if (nregions == 0 || nregions > 1024 || dir_end > cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "implausible progressive directory (%llu regions)",
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

        uint64_t want = (want_layers > 0 && (uint64_t)want_layers < nlayers)
                            ? (uint64_t)want_layers : nlayers;
        plan->want_layers = want;
        plan->partial     = (want < nlayers);

        /* The metadata region holds the per-layer bounds; the decoder needs
         * it to configure each layer's decode. Always fetch it. */
        for (uint64_t r = 0; r < nregions; r++) {
            size_t w = VOL_PROGRESSIVE_HDR_WORDS + r * VOL_SHARED_DIR_WORDS;
            uint64_t kind  = get64(hdr, w);
            uint64_t flags = get64(hdr, w + 1);
            uint64_t off   = get64(hdr, w + 2);
            uint64_t len   = get64(hdr, w + 3);
            if (off + len > cont_bytes) { ret = -1; goto done; }

            if (kind == VOL_REGION_SHARED_META) {
                if (rl.add(off, len) < 0) { ret = -1; goto done; }
            } else if (kind == VOL_REGION_LAYER && flags < want) {
                if (rl.add(off, len) < 0) { ret = -1; goto done; }
            }
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

    if (getenv("VOL_READ_PLAN_LOG"))
        fprintf(stderr,
            "[read-plan] kind=%d ranges=%zu need=%llu/%llu bytes (%.1f%%)%s\n",
            (int)plan->kind, plan->nranges,
            (unsigned long long)plan->bytes_needed,
            (unsigned long long)plan->cont_bytes,
            100.0 * (double)plan->bytes_needed / (double)plan->cont_bytes,
            plan->partial ? "  PARTIAL" : "");
    return 0;
}

herr_t
vol_container_fetch(void *under, hid_t under_vol_id, hid_t plist_id,
                    const vol_read_plan_t *plan, unsigned char **out_cbuf)
{
    if (!plan || !out_cbuf) return -1;
    *out_cbuf = NULL;

    /* calloc, not malloc: skipped regions are never touched, so on Linux they
     * are never faulted in and RSS tracks bytes_needed rather than
     * cont_bytes. Absolute offsets are preserved, so every existing decode
     * function works against this buffer unchanged. */
    unsigned char *buf = (unsigned char *)calloc(1, (size_t)plan->cont_bytes);
    if (!buf) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "out of memory allocating a %llu byte container buffer",
                (unsigned long long)plan->cont_bytes);
        return -1;
    }

    for (size_t i = 0; i < plan->nranges; i++) {
        if (vol_container_read_range(under, under_vol_id, plist_id,
                                     plan->cont_bytes,
                                     plan->ranges[i].off, plan->ranges[i].len,
                                     buf + plan->ranges[i].off) < 0) {
            free(buf);
            return -1;
        }
    }

    *out_cbuf = buf;
    return 0;
}

} /* extern "C" */