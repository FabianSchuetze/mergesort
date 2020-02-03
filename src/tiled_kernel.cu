__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C,
                                   int tile_size) {
    /* shared memory allocation */
    extern _shared_ int shareAB[];
    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];
    int C_curr = blockIdx.x * ceil((m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * ceil((m + n) / gridDim.x), (m + n));

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    _syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    _syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_lenght = B_next - B_curr;
    int total_iteration = ceil((C_length) / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (counter < total_iteration) {
        /* loading tile-size A and B elee into shared memory*/
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed)
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
        }
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_lenght - B_consumed) {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        _syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr
                                                    : C_length - C_completed;
        c_next =
            (c_next <= C_curr - C_completed) ? c_next : C_curr - C_completed;
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_lenght - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_lenght - B_consumed));
        int b_next = c_next - a_next;
        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr,
                         b_next - b_curr, C + C_curr + C_completed + c_curr);
        counter++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed += C_completed - A_consumed;
        _syncthreads();
    }
}
