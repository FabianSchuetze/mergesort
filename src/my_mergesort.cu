#include <stdio.h>
#include "../include/common.h"
#include <algorithm>
__device__ void merge(int* a, int lo, int mid, int hi, int * aux) {
    //std::cout << "merge(" << lo << ", " << mid << ", " << hi << ")\n";
    int i = lo;
    int j = mid + 1;

    for (int k = lo; k <= hi; k++) aux[k] = a[k];

    for (int k = lo; k <= hi; k++) {
        if (i > mid)
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
        int end = min((tid + 1) * width -1, size);
        int mid = start + (end - start) / 2;
        printf("Th %i, value begin %i\n", tid, src[start]);
        merge(src, start, mid, end, dest);
        printf("Th %i sort b/w %i and %i at mid %i and size %i\n", 
                tid, start, end, mid, size);
        printf("Th %i, value begin %i\n", tid, src[start]);
    }
}

//}
// float sum = size;
// int lo = tid * ceilf((sum) / (blockDim.x * gridDim.x));
// int hi = min((tid + 1) * ceilf(sum / (blockDim.x * gridDim.x)), sum);
// int mid = lo + (hi - lo) / 2;
// merge_sequential(src, m, src[mid], n, int sum, dest);
//}

void merge_sort(int* src, int* dest, int size) {
    dim3 blockDim(1);
    dim3 gridDim(3);  // ten threads, likely bug is too many selected
    //int * start = src;
    //int * aux = dest;

    // I need a good stopping criteria
    int width = 2;
    while (true) {
    //for (int width = 2; width <= size;) {
        gpu_mergesort<<<blockDim, gridDim>>>(src, dest, size, width);
        MY_CHECK(cudaDeviceSynchronize());
        printf("End of the mergesort with width %i\n", width);
        if (width == size)
            break;
        width = std::min(width*2, size);
    }
}
