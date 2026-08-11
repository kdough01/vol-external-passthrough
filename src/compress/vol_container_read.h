#ifndef VOL_CONTAINER_READ_H
#define VOL_CONTAINER_READ_H

#include <stddef.h>
#include <stdint.h>
#include "hdf5.h"
#include "metadata_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VOL_CONT_UNKNOWN = 0,
    VOL_CONT_NATIVE,
    VOL_CONT_PRESSIO,
    VOL_CONT_CHUNKED,
    VOL_CONT_SHARED,
    VOL_CONT_PROGRESSIVE
} vol_container_kind_t;

typedef struct { uint64_t off, len; } vol_range_t;

typedef struct {
    vol_container_kind_t kind;
    uint64_t     magic;
    uint64_t     cont_bytes;     /* full container size on disk            */
    uint64_t     bytes_needed;   /* sum of ranges -- the I/O actually done */
    vol_range_t *ranges;
    size_t       nranges;

    /* decode hints, filled according to kind */
    uint64_t nchunks, chunk_bytes;      /* CHUNKED / SHARED      */
    uint64_t nlayers;                   /* PROGRESSIVE           */
    unsigned want_pct;                  /* PROGRESSIVE: 1..100   */
    uint64_t pressio_chunk_elems;       /* PRESSIO               */
    int      partial;                   /* 1 if anything was skipped */
} vol_read_plan_t;

herr_t vol_container_plan(void *under, hid_t under_vol_id, hid_t plist_id,
                          const compression_ctx *ctx,
                          hid_t file_space_id, unsigned want_pct,
                          vol_read_plan_t *plan);

herr_t vol_container_fetch(void *under, hid_t under_vol_id, hid_t plist_id,
                           compression_ctx *ctx,
                           const vol_read_plan_t *plan,
                           unsigned char **out_cbuf);

void vol_read_plan_free(vol_read_plan_t *plan);

/* Read one arbitrary byte range out of the underlying 1-D byte dataset. */
herr_t vol_container_read_range(void *under, hid_t under_vol_id, hid_t plist_id,
                                uint64_t cont_bytes, uint64_t off, uint64_t len,
                                void *dst);

#ifdef __cplusplus
}
#endif
#endif /* VOL_CONTAINER_READ_H */