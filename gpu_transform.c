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
    int datatype_size; // if this is all the information we need, this can probably be removed
} datatype_ctx;

typedef struct chunking_ctx {
    int chunk_size; // I'm not sure yet what information we will need for chunking, this is just a placeholder
} chunking_ctx;

typedef struct compression_metadata {
    int library; // I need to read about libpressio to know what this struct should contain, but any information related to that function call goes here
    int compression_level;
} compression_metadata;

// any other information we want can go here
typedef struct config_params {
    int conf; // placeholder
} config_params;

typedef struct gpu_context_t {
    int device_id;
    cudaStream_t stream;
    void *d_in; // pointer to input buffer in GPU
    size_t d_in_capacity; // size of input buffer
    void *d_out;
    size_t d_out_capacity;
}

typedef struct gpu_vol_dataset_t {
    void* under_dataset;
    hid_t under_dataset;
    datatype_ctx* datatype_info;
    chunking_ctx* chunking_info;
    compression_metadata* compression_metadata;
} gpu_vol_dataset_t;

typedef struct gpu_vol_file_t {
    void* under_file;
    hid_t under_vol_id;
    gpu_context_t* gpu_ctx;
    config_params* config_params;
} gpu_vol_file_t;

// Wrapper functions to fill in structs

datatype_ctx* datatype_ctx_create()

// COMPRESSION

__global__ void gpu_compress(hid_t *d_dset[], size_t count) {
    /* 
    V1.0 this will just pass the data through the GPU and add 1 to show
    that the data is actually being passed, but not doing anything useful.
    documentation says that to perform compression, we need to use chunking,
    but that is with filters
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_dset[idx] += 1.0f;
    }
}

static herr_t
H5VL_pass_through_ext_gpu_transfer_compress(size_t nelem, hid_t dtype, void *buf[])
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
        printf("------- EXT PASS THROUGH VOL DATASET GPU TRRANSFER\n");
    #endif

    // copy dset to device
    // is mem_type_id the type of data stored???
    hid_t *d_dset;
    cudaMalloc(&d_dset, m * sizeof(hid_t));
    cudaMemcpy(d_dset, dset, m * sizeof(hid_t), cudaMemcpyHostToDevice);

    gpu_compress<<<blocks, threads>>>(d_dset, count);

    // when we compress the data, the size will be different when returned
    // need to figure out how to handle that
    cudaMemcpy(dset, d_dset, m * sizeof(hid_t), cudaMemcpyDeviceToHost);
    cudaFree(d_dset);
}

// DECOMPRESSION

__global__ void gpu_decompress(hid_t *d_dset[], size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_dset[idx] += 1.0f;
    }
}

static herr_t
H5VL_pass_through_ext_gpu_transfer_decompress(size_t count, void *dset[],
    hid_t mem_type_id[], hid_t mem_space_id[],
    hid_t file_space_id[], hid_t plist_id, void *buf[], void **req)
{

    #ifdef ENABLE_EXT_PASSTHRU_LOGGING
        printf("------- EXT PASS THROUGH VOL DATASET GPU TRRANSFER\n");
    #endif

    // copy dset to device
    // is mem_type_id the type of data stored???
    hid_t *d_dset;
    cudaMalloc(&d_dset, m * sizeof(hid_t));
    cudaMemcpy(d_dset, dset, m * sizeof(hid_t), cudaMemcpyHostToDevice);

    gpu_decompress<<<blocks, threads>>>(d_dset, count);

    cudaMemcpy(dset, d_dset, m * sizeof(hid_t), cudaMemcpyDeviceToHost);
    cudaFree(d_dset);
}