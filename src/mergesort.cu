#include "../include/merge.h"
#include "../include/common.h"
#include <sys/time.h>
#include <stdio.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
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

//__device__ void allocate_shared_memory(const int* a, int sz_a, const int* b,
// int* shared, int length) {
// int process = blockIdx.x * blockDim.x + threadIdx.x;
// int diag = process * length;
// for (int i = 0; i < length; ++i) {
// shared[i + diag] = a[i + diag];
// shared[sz_a + i + diag] = b[i + diag];
//}
////__syncthreads();
//}

__global__ void paralleMerge(const int* a, int sz_a, const int* b, int sz_b,
                             int* c, int length) {
    int process = blockIdx.x * blockDim.x + threadIdx.x;
    int diag = process * length;
    extern __shared__ int shared[];
    for (int i = 0; i < length; ++i) {
        int pos = (i + diag) / 2;
        if (blockIdx.x == 5) {
            printf("pos %i\n", pos);
        }
        shared[pos] = a[pos];
        shared[sz_a + pos] = b[pos];
    }
    __syncthreads();
    // void allocate_shared_memory(a, sz_a, b, shared, length);
    int a_start = mergepath(shared, sz_a, &shared[sz_a], sz_b, diag);
    int b_start = diag - a_start;
    merge(shared, a_start, sz_a, &shared[sz_a], b_start, sz_b, c, diag, length);
}

void cuda_merge(const int* d_A, int sz_a, const int* d_B, int sz_b, int* d_C,
                int processes) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    int size_shared =  dev_prop.sharedMemPerBlock;
    int number_blocks = (sz_a + sz_b) * sizeof(int) / size_shared + 10;
    //int size_shared = (sz_a + sz_b) * 2 * sizeof(int);
    int length = (sz_a + sz_b) / (processes * number_blocks);
    printf("the length is %i\n", length);
    dim3 blockDim(number_blocks);
    dim3 gridDim(processes);  // ten threads, likely bug is too many selected
    double beg = cpuSecond();
    paralleMerge<<<blockDim, gridDim, size_shared>>>(d_A, sz_a, d_B, sz_b, d_C,
                                                     length);
    double end = cpuSecond() - beg;
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
    printf("The operation took %.5f\n", end);
}
