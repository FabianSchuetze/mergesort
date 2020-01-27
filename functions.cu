#include <stdbool.h>
void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0; //index A
    int j = 0; // index B
    int k = 0; // index C

    while ((i< m) && (j < n)) {
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

int co_rank(int k, int*A, int m, int* B,int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k-n;
    int j_low = 0 > (k-m) ? 0: k-m;
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i-1] > B[i]) {
            delta = ((i-i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j- j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}


__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int k_curr = tid*ceil((m+n) / (blockDim.x*gridDim.x));
    int k_next = min((tid+1)*ceil((m+n) / (blockDim.x * gridDim.x)), m+n);
    int i_curr = co_rank(k_curr, A, m, B,n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], 
                     j_next-j_curr, &C[k_curr]);
}
