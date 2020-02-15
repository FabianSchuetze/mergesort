// The cuda device properties
#include <stdio.h>
int main() {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    printf("The maximum amount of threads is %i\n", dev_prop.maxThreadsPerBlock);
    printf("The maximum global memory is %zu\n", dev_prop.totalGlobalMem);
    printf("The total number of SM is %i\n", dev_prop.multiProcessorCount);
    printf("The number of register per block is %i\n", dev_prop.regsPerBlock);
    printf("The amount of shared memory per block %zu\n",
            dev_prop.sharedMemPerBlock);
    printf("The amount of shared memory per SM %zu\n",
            dev_prop.sharedMemPerMultiprocessor);

}
