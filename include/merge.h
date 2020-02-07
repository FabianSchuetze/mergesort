#ifndef merge_h
#define merge_h
void cuda_merge(int* A, int m, int* B, int n, int* C, int);
void copy_cuda(const int* A, int m, int* C);
void merge(const int* A, int m, int n, int* C);
#endif
