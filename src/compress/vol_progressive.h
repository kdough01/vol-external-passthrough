#ifndef VOL_PROGRESSIVE_H
#define VOL_PROGRESSIVE_H

#include "hdf5.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Extra bytes fetched beyond the nominal percentage. sperr_trunc_3d tolerates
 * a stream longer than it needs but fails if it is short, and the exact
 * requirement depends on SPERR's internal header layout. Over-read a little. */
#define VOL_SPERR_SLACK 4096

/* Request a reduced-fidelity read: pct is 1..100, 100 means full fidelity. */
herr_t   H5Pset_vol_progressive_pct(hid_t dxpl_id, unsigned pct);

/* Resolve the request off a DXPL. Returns 100 when unset. */
unsigned H5VL_pass_through_ext_progressive_pct(hid_t dxpl_id);

/* Percentage for chunk k. Returns the uniform want_pct when no per-chunk
 * vector has been set, pct[k] when one has. */
unsigned vol_progressive_pct_for_chunk(unsigned want_pct, uint64_t k);

/* Per-chunk fidelity. n is the chunk count; pct[i] is 1..100. */
herr_t   H5Pset_vol_progressive_pct_v(hid_t dxpl_id, size_t n,
                                      const unsigned *pct);

/* Rewrite a truncated SPERR bitstream so libpressio can decode it.
 * `stream` need only be long enough for the requested percentage.
 * *out is allocated here and must be free()d by the caller. */
herr_t   vol_sperr_truncate(const void *stream, size_t stream_len,
                            unsigned pct, void **out, size_t *out_len);

/* True when this codec supports reduced-fidelity reads. */
int      vol_codec_is_progressive(const char *compressor_id);

#ifdef __cplusplus
}
#endif
#endif /* VOL_PROGRESSIVE_H */