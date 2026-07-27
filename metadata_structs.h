#ifndef METADATA_STRUCTS_H
#define METADATA_STRUCTS_H

#include <stddef.h>
#include <stdint.h>
#include "hdf5.h"
#include <libpressio/libpressio.h>

typedef enum vol_chunking_mode {
    VOL_CHUNKING_NONE = 0,
    VOL_CHUNKING_VOL,
    VOL_CHUNKING_PRESSIO
} vol_chunking_mode_t;

/* Must come first — used by datatype_ctx */
typedef struct compression_ctx {
    char *compressor_id;
    struct pressio_options *compressor_opts;
    struct pressio *library;
    struct pressio_compressor *compressor;
    size_t ndims;
    size_t *dims;
    enum pressio_dtype dtype;
    void *compressed_buf;
    size_t compressed_chunk_size;
    uint64_t last_uncompressed_bytes;
    uint64_t last_compressed_bytes;
    void   *stage_buf;
    size_t  stage_total;
    size_t  stage_filled;
    void  *decomp_buf;
    size_t decomp_size;
    size_t read_served;
    double compress_ms;      /* device codec time (ms) from CUDA events; 0 => use caller wall clock */
    void  *stream;
    char dataset_name[256];
    int                        chunking_mode;       /* VOL_CHUNKING_* from opts_json */
    uint64_t                   chunk_mb;            /* "vol:chunk_mb", 0 = default   */
    struct pressio_compressor *chunk_wrapper;       /* cached 'chunking' meta        */
    uint64_t                   chunk_wrapper_elems; /* chunk size wrapper was built with */
} compression_ctx;

#ifdef __cplusplus
extern "C" {
#endif
size_t vol_logical_nbytes(const compression_ctx *ctx);
#ifdef __cplusplus
}
#endif

typedef struct datatype_ctx {
    void *under_obj;
    hid_t under_vol;
    compression_ctx *ctx;
    hsize_t *dims;
    int rank;
    hid_t type;
} datatype_ctx;

typedef struct chunking_ctx {
    size_t ndims;
    hsize_t *chunk_dims;
    H5D_layout_t layout;
} chunking_ctx;

typedef struct config_params {
    char *default_compression_id;
    int compression_level;
} config_params;

typedef struct gpu_vol_file_t {
    void *under_file;
    hid_t under_vol_id;
    config_params *config_params;
    int compress_on_write;
} gpu_vol_file_t;

typedef struct gpu_vol_dataset_t {
    void *under_dataset;
    hid_t under_vol_id;
    datatype_ctx *datatype_info;
    chunking_ctx *chunking_info;
    compression_ctx *comp_ctx;
    gpu_vol_file_t *file_ctx;
    int compression_requested;
} gpu_vol_dataset_t;

typedef struct {
    const unsigned char *buf;
    size_t               nbytes;
    size_t               off;
} vol_scatter_ctx;

#ifdef __cplusplus
extern "C" {
#endif
gpu_vol_dataset_t *gpu_vol_dataset_wrap(void *under_dataset,
                                        int rank, hsize_t *h5dims,
                                        hid_t type_id,
                                        enum pressio_dtype pressio_dt,
                                        hid_t dcpl_id,
                                        hid_t under_vol_id,
                                        gpu_vol_file_t *file_ctx,
                                        const char *compressor_override);
void gpu_vol_dataset_destroy(gpu_vol_dataset_t *ds_ctx);
#ifdef __cplusplus
}
#endif

#endif