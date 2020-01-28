// Merges the two arrays A and B into C; A and B are assumed to be ordered
void merge(const int* A, int m, const int* B, int n, int* C) {
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
