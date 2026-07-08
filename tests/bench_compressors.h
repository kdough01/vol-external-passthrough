#ifndef BENCH_COMPRESSORS_H
#define BENCH_COMPRESSORS_H

#include <stddef.h>
#include <stdio.h>
#include "bench_datasets.h"   /* bench_dataset_t, bench_bound_mode_t */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BENCH_CPU_CODEC = 0,
    BENCH_GPU_CODEC = 1
} bench_codec_kind_t;

typedef struct {
    const char        *name;          /* config label / dataset suffix / CSV tag */
    const char        *pressio_id;    /* compressor id for make_dcpl; NULL=default*/
    const char        *opts_json;     /* literal libpressio options JSON; NULL=none*/
    bench_codec_kind_t kind;
    int                lossless;      /* 1 => bit-exact expected                 */
    const char        *stream_opt_key;/* GPU: userptr key for the cudaStream_t   */
    const char        *note;
} bench_compressor_t;

static const bench_compressor_t BENCH_COMPRESSORS[] = {
    { "noop", "noop", NULL, BENCH_CPU_CODEC, 1, NULL,
      "Connector default (no compressor). Bit-exact baseline; isolates overhead." },

    { "cuszp", "cuszp",
      "{\"pressio:abs\": 1e-3, \"cuszp:mode_str\": \"outlier\"}",
      BENCH_GPU_CODEC, 0, "cuszp:cuda_stream",
      "GPU error-bounded lossy (A100). abs=1e-3 suits Miranda; for f32 sets "
      "consider \"pressio:rel\": 1e-3 so one config is comparable across datasets." },

    { "sz3_1e3", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-3}",
      BENCH_CPU_CODEC, 0, NULL, "CPU error-bounded lossy, abs 1e-3." },

    { "sz3_1e6", "sz3",
      "{\"sz3:error_bound_mode_str\":\"abs\",\"sz3:abs_error_bound\":1e-6}",
      BENCH_CPU_CODEC, 0, NULL, "CPU error-bounded lossy, abs 1e-6 (tighter)." },

    { "bzip2", "bzip2",
      "{\"bzip2:block_size\":9}",
      BENCH_CPU_CODEC, 1, NULL, "CPU lossless, general purpose. CPU comparator." },
};

#define BENCH_NUM_COMPRESSORS \
    ((int)(sizeof(BENCH_COMPRESSORS) / sizeof(BENCH_COMPRESSORS[0])))

static inline const char *bench_codec_kind_name(bench_codec_kind_t k) {
    return (k == BENCH_GPU_CODEC) ? "gpu" : "cpu";
}

static inline const bench_compressor_t *bench_compressor_by_name(const char *name) {
    for (int i = 0; i < BENCH_NUM_COMPRESSORS; ++i)
        if (name && 0 == __builtin_strcmp(name, BENCH_COMPRESSORS[i].name))
            return &BENCH_COMPRESSORS[i];
    return NULL;
}

/* The options JSON to hand to make_dcpl / pressio: the literal opts_json if the
 * config has one, else "{}". */
static inline const char *bench_compressor_opts(const bench_compressor_t *c) {
    return (c->opts_json && c->opts_json[0]) ? c->opts_json : "{}";
}

/* Same, but written into a buffer; if the config has no literal opts_json, fall
 * back to a generic bound generated from the DATASET (used by the pressio/filter
 * harnesses that don't take a pre-baked string). snprintf semantics. */
static inline int bench_compressor_opts_json(const bench_compressor_t *c,
                                             const bench_dataset_t *d,
                                             char *buf, size_t n) {
    if (c->opts_json && c->opts_json[0]) return snprintf(buf, n, "%s", c->opts_json);
    if (c->lossless) return snprintf(buf, n, "{}");
    const char *mode = (d->bound_mode == BENCH_BOUND_ABS) ? "pressio:abs"
                                                          : "pressio:rel";
    return snprintf(buf, n, "{\"%s\": %g}", mode, d->bound);
}

/* Emit a JSON blob describing the codec + options, to embed in the connector's
 * HDF5_VOL_CONNECTOR under_info so the VOL path uses the SAME configuration as
 * the other two harnesses. Adapt the outer key names to your connector schema. */
static inline int bench_compressor_vol_info_json(const bench_compressor_t *c,
                                                 const bench_dataset_t *d,
                                                 char *buf, size_t n) {
    char opts[256];
    bench_compressor_opts_json(c, d, opts, sizeof(opts));
    if (!c->pressio_id)
        return snprintf(buf, n, "{\"compressor\": null, \"options\": %s}", opts);
    return snprintf(buf, n,
        "{\"compressor\": \"%s\", \"options\": %s}", c->pressio_id, opts);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BENCH_COMPRESSORS_H */