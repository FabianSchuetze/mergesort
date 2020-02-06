#include <stdio.h>
#include <sys/time.h>
#include "../include/common.h"
#include "../include/merge.h"

#include "../include/merge.h"

__device__ int mergepath(const int* a, int size_a, const int* b, int size_b,
                         int diag) {
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
                      int start_b, int sz_b, int* c, int start_c, int length) {
    int i = 0;
    int j = 0;
    int k = 0;
    while (k < length) {
        if (start_a + i == sz_a)
            c[start_c + k++] = b[start_b + j++];
        else if (start_b + j == sz_b)
            c[start_c + k++] = a[start_a + i++];
        else if (a[start_a + i] <= b[start_b + j])
            c[start_c + k++] = a[start_a + i++];
        else
            c[start_c + k++] = b[start_b + j++];
    }
}

__global__ void paralleMerge(const int* a, int sz_a, const int* b, int sz_b,
                             int* c, int length) {
    /*int process = */
    int process = blockIdx.x * blockDim.x + threadIdx.x;
    int diag = process * length;
    int a_start = mergepath(a, sz_a, b, sz_b, diag);
    int b_start = diag - a_start;
    merge(a, a_start, sz_a, b, b_start, sz_b, c, diag, length);
}

__global__ void cuda_determine_range(const int* a, int sz_a, const int* b,
                                     int sz_b, int shared_size,
                                     int* block_boundaries) {
    int diag = shared_size * threadIdx.x;
    printf("the diag is %i, at thread %i with blockDim %i\n", diag, threadIdx.x,
           blockDim.x);
    if (diag < sz_a + sz_b) {
        int a_start = mergepath(a, sz_a, b, sz_b, diag);
        block_boundaries[threadIdx.x] = a_start;
        block_boundaries[threadIdx.x + blockDim.x + 1] = diag - a_start;
        printf("The range for thread %i at blockDim %i is %i\n", threadIdx.x,
               blockDim.x, a_start);
    }
}
void determine_range(const int* d_A, int sz_a, const int* d_B, int sz_b,
                     int size_shared, const int* boundaries,
                     int* d_boundaries) {
    int n_blocks = ceilf((sz_a + sz_b) / size_shared);
    cudaMalloc((void**)&d_boundaries, 2 * n_blocks * sizeof(int));
    cudaMemcpy(d_boundaries, boundaries, 2 * n_blocks * sizeof(int),
               cudaMemcpyHostToDevice);
    cuda_determine_range<<<1, gridDim>>>(d_A, sz_a, d_B, sz_b, size_shared,
                                         d_boundaries);
}
void cuda_merge(const int* d_A, int sz_a, const int* d_B, int sz_b, int* d_C,
                int length) {
    dim3 blockDim(10);
    int size_shared = 10;
    int n_blocks = ceilf((sz_a + sz_b) / size_shared);
    dim3 gridDim(n_blocks);  // ten threads, likely bug is too many selected
    int boundaries[2 * n_blocks + 2];
    int* d_boundaries;
    determine_range(d_A, sz_a, d_B, sz_b, size_shared, boundaries,
                    d_boundaries);
    // boundaries[n_blocks] = sz_a;
    // boundaries[2 * n_blocks] = sz_b;
    // cudaMemcpy(boundaries, d_boundaries, 2 * n_blocks * sizeof(int),
    // cudaMemcpyDeviceToHost);
    // MY_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < n_blocks; ++i) {
        printf("At block %i the range of a is (%i, %i) and of b is (%i, %i)\n",
               i, boundaries[i], boundaries[i + 1], boundaries[i + n_blocks],
               boundaries[i + 1 + n_blocks]);
    }
    paralleMerge<<<gridDim, blockDim>>>(d_A, sz_a, d_B, sz_b, d_C, length,
                                        d_boundaries);
}
