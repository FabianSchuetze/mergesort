
__global__
void copy(const int* d_A, int sz_a, int* d_C, const int shared_size) {
    extern __shared__ int shared[];
    int process = blockIdx.x * blockDim.x + threadIdx.x;
    /*int ele_per_block = sz_a / (shared_size*blockDim.x);*/
    for (int ph = 0; ph <= sz_a/(shared_size*blockDim.x) ; ++ph) {
        for (int i = 0; i <= shared_size / blockDim.x; ++i) {
            int pos = ph*shared_size * blockDim.x + blockIdx.x * blockDim.x +
                blockDim.x*i + threadIdx.x;
            shared[pos] = a[pos];
        }
    }
    __syncthreads();
}

void copy_cuda(const int* d_A, int sz_a, int* d_C) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    int size_shared = dev_prop.sharedMemPerBlock;
    printf("The size of shared vs a + b is: %i  vs %i", size_shared,
          (sz_a + sz_b) * sizeof(int));
    /*int number_blocks = (sz_a + sz_b) * sizeof(int) / size_shared + 10;*/
    /*//int size_shared = (sz_a + sz_b) * 2 * sizeof(int);*/
    /*int length = (sz_a + sz_b) / (processes * number_blocks);*/
    /*printf("the length is %i\n", length);*/
    dim3 blockDim(2);
    dim3 gridDim(processes);  // ten threads, likely bug is too many selected
    /*double beg = cpuSecond();*/
    copy<<<blockDim, gridDim, 10*sizeof(int)>>>(d_A, sz_a, d_C, 10);
    double end = cpuSecond() - beg;
    MY_CHECK(cudaDeviceSynchronize());
    MY_CHECK(cudaPeekAtLastError());
    printf("The operation took %.5f\n", end);
}
