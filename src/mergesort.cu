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

void cuda_merge(const int* d_A, int sz_a, const int* d_B, int sz_b, int* d_C,
                int length) {
    dim3 blockDim(1);
    dim3 gridDim(2);  // ten threads, likely bug is too many selected
    paralleMerge<<<blockDim, gridDim>>>(d_A, sz_a, d_B, sz_b, d_C, length);
}
