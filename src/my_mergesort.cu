#include <stdio.h>
//__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    //int i = 0;  // index A
    //int j = 0;  // index B
    //int k = 0;  // index C

    //while ((i < m) && (j < n)) {
        //if (A[i] <= B[j]) {
            //C[k++] = A[i++];
        //} else {
            //C[k++] = B[j++];
        //}
    //}
    //if (i == m) {
        //// done with A[] handle remaining B
        //for (; j < n; j++) {
            //C[k++] = B[j];
        //}
    //} else {
        //for (; i < m; i++) {
            //// done with B[] handle remaining A
            //C[k++] = A[i];
        //}
    //}
//}

__global__ void gpu_mergesort(int* src, int* dest, int size, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * width < size) {
        int start = tid * width;
        int end = (tid + 1) * width;
        int mid = start + (end - start) / 2;
        printf("Th %i sort b/w %i and %i at mid %d\n", tid, start, end, mid);
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

    // I need a good stopping criteria
    for (int width = 2; width < size;) {
        gpu_mergesort<<<blockDim, gridDim>>>(src, dest, size, width);
        printf("End of the mergesort with width %i\n", width);
        // change the ordering of the pointers
        width *= 2;
    }
}
