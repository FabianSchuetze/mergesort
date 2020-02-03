#include "../include/common.h"
#include <stdio.h>
#include <sys/time.h>

__device__ int binary_search(int x, int* T, int p, int r) {
    int low = p;
    int high = max(p, r+1);
    while (low < high) {
        int mid = (low + high) / 2;
        if (x <= T[mid])
            high = mid;
        else 
            low = mid + 1;
    }
    return high;
}

__global__ void pmerge(int* T, int p1, int r1, int p2, int r2, int* A, int p3) {
    int n1 = r1 - p1 + 1;
    int n2 = r2 - p2 + 1;
    if (n1 < n2) {
        int tmp_p1 = p2;
        p2 = p1;
        p1 = tmp_p1;
        int tmp_r1 = r2;
        r2 = r1;
        r1 = tmp_r1;
        int tmp_n1 = n2;
        n2 = n1;
        n1 = tmp_n1;
    }
    if (n1 == 0)
        return;
    int q1 = (p1 + r1) / 2;
    int q2 = binary_search(T[q1], T, p2, r2);
    int q3 = p3 + (q1 - p1) + (q2 - p2);
    A[q3] = T[q1];
    // spwan
    pmerge<<<1, 1>>>(T, p1, q1 -1, p2, q2 -1, A, p3);
    pmerge<<<1, 1>>>(T, q1 + 1, r1, q2, r2, A, q3 + 1);
    //cudaDeviceSynchronize();
    // sync;
}

__global__ void pmergesort(int*A , int p, int r, int* B, int s) {
    int n = r - p + 1;
    if (n == 1)
        B[s] = A[p];
    else {
        //int* T = new int[n];
        int q = (p + r) / 2;
        int qprime = q - p + 1;
        //spawn
        pmergesort<<<1, 1>>>(&A[p], 0, q -p , &T[1], 0);
        pmergesort<<<1, 1>>>(&A[q+1], 0, r - q  -1, &T[qprime + 1],0);
        //cudaDeviceSynchronize();
        //sync;
        pmerge<<<1, 1>>>(T, 1, qprime, qprime + 1, n, B, s);
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
void p_merge_sort(int* A, int size_A, int* B) {
    int* d_A;
    int* d_B;
    MY_CHECK(cudaMalloc((void**)&d_A, size_A * sizeof(int)));
    MY_CHECK(cudaMalloc((void**)&d_B, size_A * sizeof(int)));
    MY_CHECK(cudaMemcpy(d_A, A, size_A * sizeof(int), cudaMemcpyHostToDevice));
    double begin = cpuSecond();
    pmergesort<<<1, 1>>>(d_A, 0, size_A, d_B, 0);
    MY_CHECK(cudaDeviceSynchronize());
    double end = cpuSecond() - begin;
    printf("the gpu took: %.5f",end);
    MY_CHECK(cudaMemcpy(A, d_B, size_A * sizeof(int), cudaMemcpyDeviceToHost));
    //printf("the thing is (%i, %i, %i, %i, %i)\n", A[0], A[1], A[2], A[3],A[4]);
}

//int main() {
    //int* A = new int[5];
    //A[0] = 10;
    //A[1] = -10;
    //A[2] = 5;
    //A[3] = 20;
    //A[4] = -100;
    //int* d_A;
    //int* d_B;
    //MY_CHECK(cudaMalloc((void**)&d_A, 5 * sizeof(int)));
    //MY_CHECK(cudaMalloc((void**)&d_B, 5 * sizeof(int)));
    //MY_CHECK(cudaMemcpy(d_A, A, 5 * sizeof(int), cudaMemcpyHostToDevice));
    //pmergesort<<<1, 1>>>(d_A, 0, 5, d_B, 0);
    ////MY_CHECK(cudaDeviceSynchronize());
    //MY_CHECK(cudaMemcpy(A, d_B, 5 * sizeof(int), cudaMemcpyDeviceToHost));
    //printf("the thing is (%i, %i, %i, %i, %i)", A[0], A[1], A[2], A[3],A[4]);
//}
