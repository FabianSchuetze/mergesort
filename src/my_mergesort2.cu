#include <stdio.h>
#include <random>
#include <sys/time.h>
#include "../include/common.h"
#include <vector>
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

void sort(int* a, int lo, int hi, int* tmp, dim3 blockDim, dim3 gridDim) {
    if (hi <= lo) return;
    //std::cout << "sort(" << lo << ", " << hi << ")\n";
    int mid = lo + (hi - lo) / 2;
    sort(a, lo, mid, tmp, blockDim, gridDim);
    sort(a, mid + 1, hi, tmp, blockDim, gridDim);
    merge_basic_kernel<<<blockDim, gridDim>>>(&a[lo], mid-lo, &a[mid], 
                                              hi-mid, &tmp[lo]);
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
    int sz = 1000000;
    std::vector<int> a = variables(sz, -10, 10);
    printf("the thing is (%i, %i, %i, %i, %i)", a[0], a[1], a[2], a[3],a[4]);
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
    dim3 blockDim(10);
    dim3 gridDim(128);  // ten threads, likely bug is too many selected
    sort(d_A, 0, sz, d_B, blockDim, gridDim);
    double end = cpuSecond() - beg;
    printf("it took the GPU %f\n", end);
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaMemcpy(a.data(), d_B, 
                sz * sizeof(int), cudaMemcpyDeviceToHost));
    printf("the thing is (%i, %i, %i, %i, %i)", a[0], a[1], a[2], a[3],a[4]);
}

//void sort(std::vector<T>& a) {
    //std::vector<T> aux(a.size());
    //////aux.resize(a.size());
    //sort(a.data(), 0, a.size() - 1, aux.data());
//}
//template class Sedgwick<int>;
//template class Sedgwick<std::string>;
