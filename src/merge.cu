#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include "../include/merge.h"
#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0;  // index A
    int j = 0;  // index B
    int k = 0;  // index C

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        // done with A[] handle remaining B
        for (; j < n; j++) {
            C[k++] = B[j];
        }
    } else {
        for (; i < m; i++) {
            // done with B[] handle remaining A
            C[k++] = A[i];
        }
    }
}

//Identifies location in A for range of merging
__device__ void co_rank(int k, const int* A, int m, const int* B, int n, 
                        int* out) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;
    int j_low = 0 > (k - m) ? 0 : k - m;
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    out[0] = i;
}

__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = m + n;
    int k_curr = tid * ceilf((sum) / (blockDim.x * gridDim.x));
    int k_next = min((tid + 1) * ceilf(sum / (blockDim.x * gridDim.x)), sum);
    int i_curr;
    int i_next;
    co_rank(k_curr, A, m, B, n, &i_curr);
    co_rank(k_next, A, m, B, n, &i_next);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    //printf(
        //"k_curr %d, k_next %i "
        //"i_curr %i, i_next %i, j_curr %i, j_next %i, tid %d\n",
        //k_curr, k_next, i_curr, i_next, j_curr, j_next, tid);
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void cuda_merge(const int* A, int m, const int* B, int n, int* C) {
    int* d_A;
    int* d_B;
    int* d_C;
    CHECK(cudaMalloc((void**)&d_A, m * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_B, n * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_C, (m + n) * sizeof(int)));
    CHECK(cudaMemcpy(d_A, A, m * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, C, (m + n) * sizeof(int), cudaMemcpyHostToDevice));
    dim3 blockDim(4);
    dim3 gridDim(128); //ten threads, likely bug is too many selected
    double cpuStart = cpuSecond();
    merge_basic_kernel<<<blockDim, gridDim>>>(d_A, m, d_B, n, d_C);
    CHECK(cudaDeviceSynchronize());
    double cpuEnd = cpuSecond() - cpuStart;
    printf("The GPU took %.7f\n", cpuEnd);
    CHECK(cudaMemcpy(C, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));
}
