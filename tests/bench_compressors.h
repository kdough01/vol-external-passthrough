#ifndef BENCH_COMPRESSORS_H
#define BENCH_COMPRESSORS_H

#include <stddef.h>
#include <stdio.h>
#include "bench_datasets.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BENCH_CPU_CODEC = 0,
    BENCH_GPU_CODEC = 1
} bench_codec_kind_t;

typedef struct {
    const char        *name;
    const char        *pressio_id;
    bench_codec_kind_t kind;
    int                lossless;
    const char        *extra_opts;
    const char        *stream_opt_key;
    const char        *note;
} bench_compressor_t;

static const bench_compressor_t BENCH_COMPRESSORS[] = {
    { "noop",  "noop",  BENCH_CPU_CODEC, 1, NULL, NULL,
      "Lossless pass-through. Bit-exact round-trip baseline; isolates pure overhead." },

    { "bzip2", "bzip2", BENCH_CPU_CODEC, 1, NULL, NULL,
      "CPU lossless, general purpose. CPU-overhead comparator." },

    { "sz3",   "sz3",   BENCH_CPU_CODEC, 0, NULL, NULL,
      "CPU error-bounded lossy. Primary CPU comparator vs the GPU path." },

    { "cuszp", "cuszp", BENCH_GPU_CODEC, 0,
       NULL,
      "cuszp:cuda_stream",
      "GPU error-bounded lossy (A100). Needs a cudaStream_t userptr re-attached "
      "AFTER JSON options are applied (userptrs don't survive JSON)." },
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

static inline int bench_compressor_opts_json(const bench_compressor_t *c,
                                             const bench_dataset_t *d,
                                             char *buf, size_t n) {
    const char *extra = (c->extra_opts && c->extra_opts[0]) ? c->extra_opts : NULL;
    if (c->lossless) {
        if (extra) return snprintf(buf, n, "{%s}", extra);
        return snprintf(buf, n, "{}");
    }
    const char *mode = (d->bound_mode == BENCH_BOUND_ABS) ? "pressio:abs"
                                                          : "pressio:rel";
    if (extra) return snprintf(buf, n, "{\"%s\": %g, %s}", mode, d->bound, extra);
    return snprintf(buf, n, "{\"%s\": %g}", mode, d->bound);
}

static inline int bench_compressor_vol_info_json(const bench_compressor_t *c,
                                                 const bench_dataset_t *d,
                                                 char *buf, size_t n) {
    char opts[256];
    bench_compressor_opts_json(c, d, opts, sizeof(opts));
    return snprintf(buf, n,
        "{\"compressor\": \"%s\", \"options\": %s}", c->pressio_id, opts);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BENCH_COMPRESSORS_H */