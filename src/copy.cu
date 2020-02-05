#include "../include/common.h"
__device__ void copy_shared(const int* d_A, int sz_a, int* shared,
                            int shared_size, int iter, int* d_C) {
    int n_block_ele = sz_a / gridDim.x;  // elements in this block, probably
    int block_pos = n_block_ele * blockIdx.x + iter * shared_size;
    int start = threadIdx.x;
    while (start < shared_size &&
           block_pos + start < n_block_ele * (blockIdx.x + 1)) {
        int pos = block_pos + start;
        shared[pos] = d_A[pos];
        d_C[pos] = shared[pos];
        // printf("pos %i at start %i and block_pos %i\n", pos, start,
        // block_pos);
        start += blockDim.x;
    }
}
__global__ void copy(const int* d_A, int sz_a, int* d_C,
                     const int shared_size) {
    extern __shared__ int shared[];
    int n_block_ele = sz_a / gridDim.x;  // elements in this block, probably
                                         // ceil
    // printf("block_ele %i and block_idx %i\n", n_block_ele, blockIdx.x);
    for (int ph = 0; ph < ceilf((float)n_block_ele / shared_size); ++ph) {
        copy_shared(d_A, sz_a, shared, shared_size, ph, d_C);
        // int block_pos = n_block_ele * blockIdx.x + ph * shared_size;
        // int start = threadIdx.x;
        // while (start < shared_size &&
        // block_pos + start < n_block_ele * (blockIdx.x + 1)) {
        // int pos = block_pos + start;
        // printf("pos %i at start %i and block_pos %i\n", pos, start,
        // block_pos);
        // start += blockDim.x;
        //}
        //}
        __syncthreads();
    }
}

void copy_cuda(const int* d_A, int sz_a, int* d_C) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    int size_shared = dev_prop.sharedMemPerBlock;
    printf("The size of shared vs a + b is: %i  vs %i\n", size_shared,
           sz_a * sizeof(int));
    /*int number_blocks = (sz_a + sz_b) * sizeof(int) / size_shared + 10;*/
    /*//int size_shared = (sz_a + sz_b) * 2 * sizeof(int);*/
    /*int length = (sz_a + sz_b) / (processes * number_blocks);*/
    /*printf("the length is %i\n", length);*/
    dim3 blockDim(2);
    dim3 gridDim(3);  // ten threads, likely bug is too many selected
    /*double beg = cpuSecond();*/
    copy<<<blockDim, gridDim, 10 * sizeof(int)>>>(d_A, sz_a, d_C, 10);
    // double end = cpuSecond() - beg;
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
    // printf("The operation took %.5f\n", end);
}
