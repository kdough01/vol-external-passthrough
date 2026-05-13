#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

/* Public HDF5 headers */
#include "hdf5.h"

// Metadata structs -- will need to be moved to H5VLpassthru_ext.c file later, keeping here for simplicity

typedef struct datatype_ctx {
    int datatype;
    int datatype_size; // if this is all the information we need, this can probably be removed
} datatype_ctx;

typedef struct chunking_ctx {
    int chunk_size; // I'm not sure yet what information we will need for chunking, this is just a placeholder
    int chunk_dims[2]; // how should we specify dimensions, I put in an array here but idk if that's the best way
    int layout_type;
    int mapping;
} chunking_ctx;

typedef struct compression_ctx {
    int block_sizes;
    int compressed_chunk_size;
} compression_ctx;

// any other static information we want can go here
typedef struct config_params {
    int device_id;
    int min_size_for_gpu;
    int max_device_memory_bytes;
    int compression_library;
    int compression_level;
} config_params;

typedef struct gpu_context_t {
    int device_id;
    cudaStream_t stream;
    void *d_in; // pointer to input buffer in GPU
    size_t d_in_capacity; // size of input buffer
    void *d_out;
    size_t d_out_capacity;
} gpu_context_t;

typedef struct gpu_vol_dataset_t {
    void* under_dataset;
    hid_t under_vol_id;
    datatype_ctx* datatype_info;
    chunking_ctx* chunking_info;
    compression_ctx* comp_ctx;
} gpu_vol_dataset_t;

typedef struct gpu_vol_file_t {
    void* under_file;
    hid_t under_vol_id;
    gpu_context_t* gpu_ctx;
    config_params* config_params;
} gpu_vol_file_t;

// Wrapper functions to fill in structs

config_params* config_params_create(hid_t fapl_id)
{
    (void)fapl_id;

    config_params *p = (config_params*)calloc(1, sizeof(config_params));
    if (!p) {
        return NULL;
    }

    p->device_id = 0;
    p->min_size_for_gpu = 256 * 1024;
    p->max_device_memory_bytes = 2ULL * 1024 * 1024 * 1024;
    p->compression_library = 0;
    p->compression_level = 1;

    return p;
}

gpu_context_t* gpu_context_create(config_params *conf_params)
{
    gpu_context_t *gpu_ctx = (gpu_context_t*)calloc(1, sizeof(gpu_context_t));

    gpu_ctx->device_id = conf_params->device_id;
    cudaSetDevice(gpu_ctx->device_id);

    cudaStreamCreate(&gpu_ctx->stream);

    gpu_ctx->d_in_capacity = conf_params->max_device_memory_bytes;
    gpu_ctx->d_out_capacity = conf_params->max_device_memory_bytes;

    cudaMalloc(&gpu_ctx->d_in, gpu_ctx->d_in_capacity);
    cudaMalloc(&gpu_ctx->d_out, gpu_ctx->d_out_capacity);

    return gpu_ctx;
}

datatype_ctx* datatype_ctx_create(hid_t dataset_id)
{
    datatype_ctx *dt_ctx = (datatype_ctx*)calloc(1, sizeof(datatype_ctx));

    hid_t dtype = H5Dget_type(dataset_id);

    dt_ctx->datatype_size = H5Tget_size(dtype);
    dt_ctx->datatype = H5Tget_class(dtype);

    H5Tclose(dtype);

    return dt_ctx;
}

chunking_ctx* chunking_ctx_create(hid_t dataset_id)
{
    chunking_ctx *chunk_ctx = (chunking_ctx*)calloc(1, sizeof(chunking_ctx));

    hid_t space = H5Dget_space(dataset_id);
    hid_t dcpl = H5Dget_create_plist(dataset_id);

    chunk_ctx->layout_type = H5Dget_layout(dataset_id);

    if (chunk_ctx->layout_type == H5D_CHUNKED) {
        H5Pget_chunk(dcpl, 2, chunk_ctx->chunk_dims);
    }

    chunk_ctx->chunk_size = H5Sget_simple_extent_npoints(space);

    H5Pclose(dcpl);
    H5Sclose(space);

    return chunk_ctx;
}

compression_ctx* compression_ctx_create(hid_t dataset_id)
{
    compression_ctx *comp_ctx = (compression_ctx*)calloc(1, sizeof(compression_ctx));
    
    comp_ctx->block_sizes = 0;
    comp_ctx->compressed_chunk_size = 0;

    return comp_ctx;
}

gpu_vol_dataset_t* gpu_vol_dataset_wrap(void *under_dataset, hid_t dataset_id, hid_t under_vol_id)
{
    gpu_vol_dataset_t *gpu_dataset_ctx = (gpu_vol_dataset_t*)calloc(1, sizeof(gpu_vol_dataset_t));

    gpu_dataset_ctx->under_dataset = under_dataset;
    gpu_dataset_ctx->under_vol_id = under_vol_id;
    gpu_dataset_ctx->datatype_info = datatype_ctx_create(dataset_id);
    gpu_dataset_ctx->chunking_info = chunking_ctx_create(dataset_id);
    gpu_dataset_ctx->comp_ctx = compression_ctx_create(dataset_id);

    return gpu_dataset_ctx;
}

gpu_vol_file_t* gpu_vol_file_wrap(hid_t fapl_id, hid_t under_vol_id, void *under_file)
{
    gpu_vol_file_t *gpu_vol_file_ctx = (gpu_vol_file_t*)calloc(1, sizeof(gpu_vol_file_t));

    gpu_vol_file_ctx->under_file = under_file;
    gpu_vol_file_ctx->under_vol_id = under_vol_id;

    gpu_vol_file_ctx->config_params = config_params_create(fapl_id);
    gpu_vol_file_ctx->gpu_ctx = gpu_context_create(gpu_vol_file_ctx->config_params);

    return gpu_vol_file_ctx;
}

// COMPRESSION
template <typename T>
__global__ void gpu_compress(T* d_dset, size_t count) {
    /* 
    V1.0 this will just pass the data through the GPU and add 1 to show
    that the data is actually being passed, but not doing anything useful.
    documentation says that to perform compression, we need to use chunking,
    but that is with filters
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_dset[idx] += 1;
    }
}

herr_t
H5VL_pass_through_ext_gpu_transfer_compress(size_t nelem, hid_t dtype, void *buf)
{

    /* 
    
    Arguments we need:
    - size_t - nelem - number of elements
    - hid_t - dtype - type of data elements
    - layout of data?
    - file_space_id - specifies where the data lands in the dataset
    - void - buf - pointer to the actual data

    I don't quite understand the asynchronous calls yet, so I am temporarily
    ignoring them. I am also just assuming a simple layout for this V1 and will
    worry about that in V1.1

    */


    /* 
    I think at least initially, we can have the args be exactly the same as the
    dataset read. I see these functions as similar - we're "reading" the dset to
    the gpu (I know it's not strictly reading, but there is similarity)

    the way I am thinking about this is:
        compression = reading
        decompression = writing
    */

    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("GPU TRANSFORM CALLED: nelem=%zu dtype=%d\n", nelem, dtype);
    #endif

    // copy dset to device
    // is mem_type_id the type of data stored???
    void *d_dset;
    H5T_class_t cls = H5Tget_class(dtype);
    size_t size = H5Tget_size(dtype);
    size_t bytes = nelem * H5Tget_size(dtype);

    cudaMalloc(&d_dset, bytes);
    cudaMemcpy(d_dset, buf[0], bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (nelem + threads - 1) / threads;

    switch (cls) {
        case H5T_INTEGER:
            if (size==4) {
                // printf("GPU INT TRANSFORM CALLED: nelem=%zu dtype=%ld\n", nelem, dtype);
                gpu_compress<int><<<blocks, threads>>>((int*)d_dset, nelem);
            }
            break;
        case H5T_FLOAT:
            if (size==4) {
                // printf("GPU FLOAT TRANSFORM CALLED: nelem=%zu dtype=%d\n", nelem, dtype);
                gpu_compress<float><<<blocks, threads>>>((float*)d_dset, nelem);
            }
            break;
        default:

    }
    cudaDeviceSynchronize();

    // when we compress the data, the size will be different when returned
    // need to figure out how to handle that
    cudaMemcpy(buf[0], d_dset, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dset);

    return 0;
}

// DECOMPRESSION

__global__ void gpu_decompress(float *d_dset, size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < count) {
        d_dset[idx] += 1.0f;
    }
}

static herr_t
H5VL_pass_through_ext_gpu_transfer_decompress(size_t count, void *dset[], void *buf[])
{

    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("GPU TRANSFORM CALLED: nelem=%zu dtype=%d\n", nelem, dtype);
    #endif

    // copy dset to device
    // is mem_type_id the type of data stored???

    (void)dset;

    size_t bytes = count * sizeof(float);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    float *d_dset;
    cudaMalloc(&d_dset, bytes);
    cudaMemcpy(d_dset, buf[0], bytes, cudaMemcpyHostToDevice);

    gpu_decompress<<<blocks, threads>>>(d_dset, count);
    cudaDeviceSynchronize();

    cudaMemcpy(buf[0], d_dset, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dset);

    return 0;
}