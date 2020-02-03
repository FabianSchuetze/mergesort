#include <stdio.h>
#include "../include/common.h"
#include <vector>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include <cuda_profiler_api.h>
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

std::vector<int> variables(int size, int min, int max) {
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(min,max); // guaranteed unbiased
    std::vector<int> res(size);
    for (int i = 0; i < size; ++i) {
        res[i] = uni(rng);
    }
    return res;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
int main() {
    int sz = 10000;
    std::vector<int> a = variables(sz, -1000, 1000);
    //printf("the thing is (%i, %i, %i, %i, %i)", a[0], a[1], a[2], a[3],a[4]);
    //std::vector<int> a(5);
    //int* A = new int[5];
    //A[0] = 10;
    //A[1] = -10;
    //A[2] = 5;
    //A[3] = 20;
    //A[4] = -100;
    int* d_A;
    int* d_B;
    MY_CHECK(cudaMalloc((void**)&d_A, sz * sizeof(int)));
    MY_CHECK(cudaMalloc((void**)&d_B, sz * sizeof(int)));
    MY_CHECK(cudaMemcpy(d_A, a.data(), 
                sz * sizeof(int), cudaMemcpyHostToDevice));
    double beg = cpuSecond();
    merge_sort(d_A, d_B, sz);
    MY_CHECK(cudaDeviceSynchronize());
    double end = cpuSecond() - beg;
    //printf("it took the GPU %f\n", end);
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaMemcpy(a.data(), d_B, 
                sz * sizeof(int), cudaMemcpyDeviceToHost));
    //printf("the thing is (%i, %i, %i, %i, %i)", a[0], a[1], a[2], a[3],a[4]);
}
