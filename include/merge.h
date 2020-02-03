#ifndef merge_h
#define merge_h
void cuda_merge(const int* A, int m, const int* B, int n, int* C);
void merge(const int* A, int m, int n, int* C);
#endif
