#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

/* Public HDF5 headers */
#include "hdf5.h"

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