#ifndef VOL_SHARED_META_H
#define VOL_SHARED_META_H

#include <stddef.h>
#include <stdint.h>
#include "hdf5.h"
#include "metadata_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* "VOLSHR01" */
#define VOL_SHARED_MAGIC        ((uint64_t)0x564F4C5348523031ULL)
#define VOL_SHARED_VERSION      ((uint64_t)1)
#define VOL_SHARED_HDR_WORDS    8
#define VOL_SHARED_DIR_WORDS    4      /* kind, flags, offset, length */

/* Region kinds. */
#define VOL_REGION_SHARED_META  1u   /* once-per-dataset codec metadata      */
#define VOL_REGION_CHUNK_TABLE  2u   /* uint64 csize[nchunks]                */
#define VOL_REGION_PAYLOAD      3u   /* concatenated per-chunk payloads      */
#define VOL_REGION_LAYER        4u   /* RESERVED: progressive layer, flags=ix */

typedef struct vol_shared_provider {
    const char *name;
    int  (*can_share)(const compression_ctx *ctx);
    int  (*derive)(compression_ctx *ctx, const void *data, size_t nbytes,
                   size_t chunk_bytes, void **out_meta, size_t *out_len);
    int  (*attach)(compression_ctx *ctx, const void *meta, size_t len);
    int  (*compress_chunk)(compression_ctx *ctx, const void *in, size_t nbytes,
                           void **out_cbuf, uint64_t *out_csize);
    int  (*decompress_chunk)(compression_ctx *ctx, const void *cbuf, size_t csize,
                             void *out, size_t out_bytes);
    void (*release)(compression_ctx *ctx);
} vol_shared_provider;

const vol_shared_provider *vol_shared_provider_for(const compression_ctx *ctx);

const char *vol_shared_providers_list(void);

herr_t H5VL_pass_through_ext_compress_shared(compression_ctx *ctx,
                                             const void *data, size_t nbytes,
                                             size_t chunk_bytes_req,
                                             void **out_container,
                                             uint64_t *out_total);

herr_t H5VL_pass_through_ext_decompress_shared(compression_ctx *ctx,
                                               const void *cbuf, size_t cont_bytes,
                                               void *out, size_t out_bytes);

herr_t H5VL_pass_through_ext_shared_probe(const void *cbuf, size_t cont_bytes,
                                          uint64_t *out_nchunks,
                                          uint64_t *out_chunk_bytes,
                                          uint64_t *out_total_bytes,
                                          uint64_t *out_shared_len);

#ifdef __cplusplus
}
#endif
#endif /* VOL_SHARED_META_H */