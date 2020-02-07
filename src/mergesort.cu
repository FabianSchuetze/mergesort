#include <stdio.h>
#include <sys/time.h>
#include "../include/common.h"
#include "../include/merge.h"

#include "../include/merge.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
__device__ int mergepath(int* a, int size_a, int* b, int size_b, int diag) {
    if (diag == 0) return 0;
    int begin = max(0, diag - size_b);
    int end = min(diag, size_a);

    while (begin < end) {
        int mid = (begin + end) >> 1;
        int a_val = a[mid];
        int b_val = b[diag - 1 - mid];
        bool pred = a_val < b_val;
        if (pred)
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

__device__ void merge(const int* a, int start_a, int sz_a, const int* b,
                      int start_b, int sz_b, int* c, int start_c, int length,
                      int* ranges) {
    int i = 0;
    int j = 0;
    int k = 0;
    int pos = threadIdx.x*length;
    //printf("For threadix %i at blockDim %i the pos is %i\n",
           //threadIdx.x, blockIdx.x, pos); 
    // int val;
    while (k < length && pos < sz_a + sz_b) {
        if (start_a + i == sz_a)
            c[start_c + k++] = b[start_b + j++];
        else if (start_b + j == sz_b) {
            c[start_c + k++] = a[start_a + i++];
        } else if (a[start_a + i] <= b[start_b + j])
            c[start_c + k++] = a[start_a + i++];
        else
            c[start_c + k++] = b[start_b + j++];
        //printf("The value of position %i is %i for thread %i at block %i\n",
               //start_c + k - 1, c[start_c + k - 1], threadIdx.x, blockIdx.x);
        pos++;
    }
}

__device__ void loadtodevice(const int* a, int sz_a, const int* b, int sz_b,
                             int* boundaries, int* shared) {
    int tid = threadIdx.x;
    while (tid < boundaries[2]) {
        shared[tid] = a[tid + boundaries[0]];
        tid += blockDim.x;
    }
    tid = threadIdx.x;
    while (tid < boundaries[3]) {
        shared[boundaries[2] + tid] = b[tid + boundaries[1]];
        tid += blockDim.x;
    }
}

__device__ void devicetoglobal(const int* shared, int sz_a, int sz_b,
                               const int* boundaries, int* c) {
    int start_block_a = boundaries[blockIdx.x];
    int end_block_a =
        (gridDim.x - 1 > blockIdx.x) ? boundaries[blockIdx.x + 1] : sz_a;
    int start_a = threadIdx.x;
    while (start_block_a + start_a < end_block_a) {
        c[start_a + start_block_a] = shared[start_a];
        start_a += blockDim.x;
    }
    int start_block = boundaries[blockIdx.x + gridDim.x];
    start_a = threadIdx.x;
    int end_block = (gridDim.x - 1 > blockIdx.x)
                        ? boundaries[blockIdx.x + gridDim.x + 1]
                        : sz_b;
    while (start_block + start_a < end_block) {
        c[sz_a + start_a + start_block] =
            shared[end_block_a - start_block_a + start_a];
        start_a += blockDim.x;
    }
}
__device__ void ranges(int* ranges, int sz_a, int sz_b, int* boundaries) {
    if (threadIdx.x == 0) {
        ranges[0] = boundaries[2 * blockIdx.x];
        ranges[1] = boundaries[2 * blockIdx.x + 1];
        int pos = 2 * (blockIdx.x + 1);
        ranges[2] = boundaries[pos] - ranges[0];
        ranges[3] = boundaries[pos + 1] - ranges[1];
    }
}

__global__ void paralleMerge(const int* a, int sz_a, const int* b, int sz_b,
                             int* c, int* boundaries, int length,
                             int size_shared) {
    extern __shared__ int shared[];
    __shared__ int block_ranges[4];
    ranges(block_ranges, sz_a, sz_b, boundaries);
    __syncthreads();
    loadtodevice(a, sz_a, b, sz_b, block_ranges, shared);
    __syncthreads();
    int process = threadIdx.x;
    int diag = process * length;
    if (diag < block_ranges[2] + block_ranges[3]) {
        int a_start =
            mergepath(shared, block_ranges[2], &shared[block_ranges[2]],
                      block_ranges[3], diag);
        int b_start = diag - a_start;
        merge(shared, a_start, block_ranges[2], &shared[block_ranges[2]],
              b_start, block_ranges[3], c, diag + blockIdx.x * size_shared,
              length, block_ranges);
    }
}

__global__ void determine_range(int* a, int sz_a, int* b, int sz_b,
                                int shared_size, int* block_boundaries) {
    int diag = shared_size * threadIdx.x;
    if (diag < sz_a + sz_b) {
        int a_start = mergepath(a, sz_a, b, sz_b, diag);
        block_boundaries[2 * threadIdx.x] = a_start;
        block_boundaries[2 * threadIdx.x + 1] = diag - a_start;
    }
    // don't know why this works but the previous didn't
    if (threadIdx.x == 0) {
        block_boundaries[2 * blockDim.x] = sz_a;
        block_boundaries[2 * blockDim.x + 1] = sz_b;
    }
}

void cuda_merge(int* d_A, int sz_a, int* d_B, int sz_b, int* d_C, int length) {
    dim3 blockDim(128);
    int size_shared = 500;
    int n_blocks = ceilf((float)(sz_a + sz_b) / size_shared);
    printf("number of blocks %i and sz_b %i\n", n_blocks, sz_b);
    dim3 gridDim(n_blocks);
    int boundaries[2 * n_blocks + 2];
    int* d_boundaries;
    cudaMalloc((void**)&d_boundaries, 2 + 2 * n_blocks * sizeof(int));
    cudaMemcpy(d_boundaries, boundaries, 2 + 2 * n_blocks * sizeof(int),
               cudaMemcpyHostToDevice);
    double beg = cpuSecond();
    determine_range<<<1, gridDim>>>(d_A, sz_a, d_B, sz_b, size_shared,
                                    d_boundaries);
    //MY_CHECK(cudaDeviceSynchronize());
    paralleMerge<<<gridDim, blockDim, size_shared * sizeof(int)>>>(
        d_A, sz_a, d_B, sz_b, d_C, d_boundaries, 3, size_shared);
    MY_CHECK(cudaDeviceSynchronize());
    double end = cpuSecond() - beg;
    printf("The GPU took: %f\n", end);
}
