#ifndef VOL_TYPES_H
#define VOL_TYPES_H

#include <stddef.h>
#include <libpressio/libpressio.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

typedef struct compression_ctx {
    char *compressor_id;
    struct pressio_options *compressor_opts;
    struct pressio *library;
    struct pressio_compressor *compressor;
    size_t ndims; // replace with std::vector
    size_t *dims; // replace with std::vector
    enum pressio_dtype dtype; // replace with pressio dtype
    void *compressed_buf;
    size_t compressed_chunk_size;
    uint64_t last_uncompressed_bytes;
    uint64_t last_compressed_bytes;
} compression_ctx;

typedef struct gpu_context_t {
    int device_id;
#ifdef USE_CUDA
    cudaStream_t stream;
    void *d_in; // pointer to input buffer in GPU
    size_t d_in_capacity; // size of input buffer
    void *d_out;
    size_t d_out_capacity;
#endif
} gpu_context_t;

typedef struct gpu_vol_file_t {
    void* under_file;
    hid_t under_vol_id;
    gpu_context_t* gpu_ctx;
    config_params* config_params;
} gpu_vol_file_t;

typedef struct gpu_vol_dataset_t {
    void* under_dataset;
    hid_t under_vol_id;
    datatype_ctx* datatype_info;
    chunking_ctx* chunking_info;
    compression_ctx* comp_ctx;
    gpu_vol_file_t* file_ctx;
} gpu_vol_dataset_t;

#endif