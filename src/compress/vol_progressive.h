/* =========================================================================
 *   word 0  VOL_PROGRESSIVE_MAGIC
 *   word 1  VOL_PROGRESSIVE_VERSION
 *   word 2  nregions            (= nlayers + 1)
 *   word 3  nlayers
 *   word 4  elem_size           (bytes per element, cross-check)
 *   word 5  total_bytes         (logical dataset size)
 *   word 6  reserved (0)
 *   word 7  reserved (0)
 *   then nregions x [kind][flags][offset][length]
 *        kind=VOL_REGION_SHARED_META, flags=0        -> bounds + global stats
 *        kind=VOL_REGION_LAYER,       flags=k        -> layer k payload
 *   then the region payloads.
 *
 * The SHARED_META region is:  double bounds[nlayers]; double gmin; double gmax;
 * ========================================================================= */
#ifndef VOL_PROGRESSIVE_H
#define VOL_PROGRESSIVE_H

#include <stddef.h>
#include <stdint.h>
#include "hdf5.h"
#include "metadata_structs.h"
#include "vol_shared_meta.h"     /* region kinds + directory word layout */

#ifdef __cplusplus
extern "C" {
#endif

/* "VOLPRG01" */
#define VOL_PROGRESSIVE_MAGIC     ((uint64_t)0x564F4C5052473031ULL)
#define VOL_PROGRESSIVE_VERSION   ((uint64_t)1)
#define VOL_PROGRESSIVE_HDR_WORDS 8

#ifndef VOL_CHUNKING_PROGRESSIVE
#define VOL_CHUNKING_PROGRESSIVE  4
#endif

#define VOL_PROGRESSIVE_DEFAULT_LAYERS 4
#define VOL_PROGRESSIVE_DEFAULT_RATIO  8.0
#define VOL_PROGRESSIVE_MAX_LAYERS     16

herr_t H5VL_pass_through_ext_compress_progressive(compression_ctx *ctx,
                                                  const void *data, size_t nbytes,
                                                  int nlayers, double ratio,
                                                  void **out_container,
                                                  uint64_t *out_total);

herr_t H5VL_pass_through_ext_decompress_progressive(compression_ctx *ctx,
                                                    const void *cbuf,
                                                    size_t cont_bytes,
                                                    void *out, size_t out_bytes,
                                                    int want_layers,
                                                    double *out_achieved_bound);

/* Geometry probe without decoding. */
herr_t H5VL_pass_through_ext_progressive_probe(const void *cbuf, size_t cont_bytes,
                                               uint64_t *out_nlayers,
                                               uint64_t *out_total_bytes);

int H5VL_pass_through_ext_progressive_want(hid_t dxpl_id);

/* Layer count / ratio for the WRITE side, from ctx options then environment. */
int    H5VL_pass_through_ext_progressive_layers(const compression_ctx *ctx);
double H5VL_pass_through_ext_progressive_ratio(const compression_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif /* VOL_PROGRESSIVE_H */