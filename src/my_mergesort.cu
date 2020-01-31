#include <stdio.h>
#include "../include/common.h"
#include <algorithm>
__device__ void merge(int* a, int lo, int mid, int hi, int * aux) {
    int i = lo;
    int j = mid;

    for (int k = lo; k <= hi; k++) aux[k] = a[k];

    for (int k = lo; k <= hi; k++) {
        if (i >= mid)
            a[k] = aux[j++];
        else if (j > hi)
            a[k] = aux[i++];
        else if (aux[j] < aux[i])
            a[k] = aux[j++];
        else
            a[k] = aux[i++];
    }
}

__global__ void gpu_mergesort(int* src, int* dest, int size, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * width < size) {
        int start = tid * width;
        int end = min((tid + 1) * width, size) -1;
        int mid = min(start + (width >> 1), size -1);
        //printf("Th %i, value begin %i\n", tid, src[start]);
        merge(src, start, mid, end, dest);
        //printf("Th %i sort b/w %i and %i at mid %i and size %i\n", 
                //tid, start, end, mid, size);
        //printf("Th %i, value begin %i\n", tid, src[start]);
    }
}

void merge_sort(int* src, int* dest, int size) {
    dim3 blockDim(10);
    dim3 gridDim(128);  // ten threads, likely bug is too many selected
    int width = 2;
    while (true) {
        gpu_mergesort<<<blockDim, gridDim>>>(src, dest, size, width);
        //MY_CHECK(cudaDeviceSynchronize());
        //printf("End of the mergesort with width %i\n", width);
        if (width > size)
            break;
        width *= 2;
    }
}
