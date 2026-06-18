#ifndef METADATA_STRUCTS_H
#define METADATA_STRUCTS_H

#include <stddef.h>
#include <stdint.h>
#include "hdf5.h"
#include <libpressio/libpressio.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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
} compression_ctx;

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
    int device_id;
    size_t min_size_for_gpu;
    size_t max_device_memory_bytes;
    char *default_compression_id;
    int compression_level;
} config_params;

typedef struct gpu_context_t {
    int device_id;
#ifdef USE_CUDA
    cudaStream_t stream;
    void *d_in;
    size_t d_in_capacity;
    void *d_out;
    size_t d_out_capacity;
#endif
} gpu_context_t;

typedef struct gpu_vol_file_t {
    void *under_file;
    hid_t under_vol_id;
    gpu_context_t *gpu_ctx;
    config_params *config_params;
} gpu_vol_file_t;

typedef struct gpu_vol_dataset_t {
    void *under_dataset;
    hid_t under_vol_id;
    datatype_ctx *datatype_info;
    chunking_ctx *chunking_info;
    compression_ctx *comp_ctx;
    gpu_vol_file_t *file_ctx;
    gpu_context_t *gpu_ctx;
    int compression_requested;
} gpu_vol_dataset_t;

static inline const char* gpu_stream_key(const char* id) {
    if (strncmp(id, "nvcomp", 6) == 0) return "nvcomp:stream";
    if (strncmp(id, "cuszp",  5) == 0) return "cuszp:stream";
    if (strncmp(id, "cusz",   4) == 0) return "cusz:stream";
    return NULL;
}

#endif