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

// Identifies location in A for range of merging
__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    // int* out) {
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
    return i;
    // out[0] = i;
}

__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C,
                                   int tile_size) {
    /* shared memory allocation */
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];
    int C_curr = blockIdx.x * ceilf((m + n) / gridDim.x);
    int tmp = (blockIdx.x + 1) * ceilf((m + n) / gridDim.x);
    int C_next = min(tmp, (m + n));
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();
    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_lenght = B_next - B_curr;
    //printf("C_length %i\n", C_length);
    int total_iteration = ceilf((float( C_length)) / tile_size);
    //printf("total_iteration %i\n" , total_iteration);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (counter < total_iteration) {
        //printf("inside \n");
        /* loading tile-size A and B elee into shared memory*/
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                //int pos = A_curr + A_consumed + i + threadIdx.x;
                //if (pos > n) {
                    //printf("pos %i\n, thread idx %i", pos, threadIdx.x);
                //}
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_lenght - B_consumed) {
                //int pos = B_curr + B_consumed + i + threadIdx.x;
                //printf("pos %i\n, thread idx %i", pos, threadIdx.x);
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();
        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr
                                                    : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next
                                                    : C_length - C_completed;
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_lenght - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_lenght - B_consumed));
        int b_next = c_next - a_next;
        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr,
                         b_next - b_curr, C + C_curr + C_completed + c_curr);
        //printf("THe index if %i",  C_curr + C_completed + c_curr);
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

//__global__ void merge_basic_kernel(int* A, int m, int n, int* C) {
// int tid = blockIdx.x * blockDim.x + threadIdx.x;
// float sum = m + n;
// int k_curr = tid * ceilf((sum) / (blockDim.x * gridDim.x));
// int k_next = min((tid + 1) * ceilf(sum / (blockDim.x * gridDim.x)), sum);
////int i_curr;
////int i_next;
// int i_curr = co_rank(k_curr, A, m, &A[m], n);
// int i_next = co_rank(k_next, A, m, &A[m], n);
// int j_curr = k_curr - i_curr;
// int j_next = k_next - i_next;
//// printf(
////"k_curr %d, k_next %i "
////"i_curr %i, i_next %i, j_curr %i, j_next %i, tid %d\n",
//// k_curr, k_next, i_curr, i_next, j_curr, j_next, tid);
// merge_sequential(&A[i_curr], i_next - i_curr, &A[m + j_curr],
// j_next - j_curr, &C[k_curr]);
//}

__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = m + n;
    int k_curr = tid * ceilf((sum) / (blockDim.x * gridDim.x));
    int k_next = min((tid + 1) * ceilf(sum / (blockDim.x * gridDim.x)), sum);
    // int i_curr;
    // int i_next;
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    // printf(
    //"k_curr %d, k_next %i "
    //"i_curr %i, i_next %i, j_curr %i, j_next %i, tid %d\n",
    // k_curr, k_next, i_curr, i_next, j_curr, j_next, tid);
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

double cpuSecond3() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
// void cuda_merge(const int* A, int m, int n, int* C) {
// int* d_A;
// int* d_C;
// CHECK(cudaMalloc((void**)&d_A, (m + n) * sizeof(int)));
// CHECK(cudaMalloc((void**)&d_C, (m + n) * sizeof(int)));
// CHECK(cudaMemcpy(d_A, A, (m + n) * sizeof(int), cudaMemcpyHostToDevice));
// CHECK(cudaMemcpy(d_C, C, (m + n) * sizeof(int), cudaMemcpyHostToDevice));
// dim3 blockDim(100);
// dim3 gridDim(128);  // ten threads, likely bug is too many selected
// double cpuStart = cpuSecond3();
// int width = std::pow(2, 15);
// for (int i = 0; i < m + n;) {
// merge_basic_kernel<<<blockDim, gridDim>>>(&d_A[i], width, width,
//&d_C[i]);
// i += width * 2;
// printf("iter number %i\n", i);
//}
// merge_basic_kernel<<<blockDim, gridDim>>>(d_C, 2 * width, 2 * width, d_A);
// CHECK(cudaDeviceSynchronize());
// double cpuEnd = cpuSecond3() - cpuStart;
// printf("The GPU took %.7f\n", cpuEnd);
// CHECK(cudaMemcpy(C, d_A, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));
//}

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
    dim3 blockDim(128);
    dim3 gridDim(16);  // ten threads, likely bug is too many selected
    double cpuStart = cpuSecond3();
    int shared_memory_size = 2 * 1024 * sizeof(int);
    //merge_tiled_kernel<<<blockDim, gridDim, shared_memory_size>>>(d_A, m, d_B,
                                                                  //n, d_C, 1024);
    merge_basic_kernel<<<blockDim, gridDim>>>(d_A, m, d_B, n, d_C);
    CHECK(cudaDeviceSynchronize());
    double cpuEnd = cpuSecond3() - cpuStart;
    printf("The GPU took %.7f\n", cpuEnd);
    CHECK(cudaMemcpy(C, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));
}
