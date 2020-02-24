__global__ void square(int *array, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        array[tid] = array[tid] * array[tid];
    }
}

void fill_array(int* array, int n) {
    for (int i = 0; i < n; ++i) array[i] = i;
}

int main() {
    int sz = 3;
    int a[sz];
    fill_array(a, sz);
    // 1. Initialize Memory on GPU and copy from RAM//
    int* d_a;
    cudaMalloc((int**)&d_a, sz * sizeof(int));
    cudaMemcpy(d_a, a, sz * sizeof(int), cudaMemcpyHostToDevice);
    // 2. Define Thread Structure and Launch Kernel //
    dim3 block(4);
    dim3 grid(2);
    square<<<grid, block>>>(d_a, sz);
    // 2. Copy memory back to RAM //
    cudaMemcpy(a, d_a, sz * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << a[2] << std::endl;
}
