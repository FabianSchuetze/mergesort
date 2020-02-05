#include "../include/common.h"
__device__ void copy_shared(const int* d_A, int sz_a, int* shared,
                            int shared_size, int iter, int* d_C) {
    int n_block_ele = ceil((float)sz_a / gridDim.x);
    int block_pos = n_block_ele * blockIdx.x + iter * shared_size;
    int start = threadIdx.x;
    while (start < shared_size &&
           block_pos + start < n_block_ele * (blockIdx.x + 1)) {
        shared[start] = d_A[block_pos + start];
        d_C[block_pos + start] = shared[start];
        start += blockDim.x;
    }
}
__global__ void copy(const int* d_A, int sz_a, int* d_C, const int shared_sz) {
    extern __shared__ int shared[];
    for (int ph = 0; ph < ceilf((float)sz_a / (gridDim.x * shared_sz)); ++ph) {
        copy_shared(d_A, sz_a, shared, shared_sz, ph, d_C);
        __syncthreads();
    }
}

void copy_cuda(const int* d_A, int sz_a, int* d_C) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    //int size_shared = dev_prop.sharedMemPerBlock;
    //printf("The size of shared vs a + b is: %i  vs %i\n", size_shared,
           //sz_a * sizeof(int));
    dim3 blockDim(10);
    dim3 gridDim(100);  // ten threads, likely bug is too many selected
    int size_shared = 10000;
    /*double beg = cpuSecond();*/
    copy<<<gridDim, blockDim,  size_shared* sizeof(int)>>>(d_A, sz_a, d_C, 
                                                           size_shared);
    // double end = cpuSecond() - beg;
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
    // printf("The operation took %.5f\n", end);
}
