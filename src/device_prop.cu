// The cuda device properties
#include <stdio.h>
int main() {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    printf("Comput cappbility %i and %i\n", dev_prop.major, dev_prop.minor);
    printf("SP count %i\n", dev_prop.multiProcessorCount);
    printf("The maximum amount of threads per Block %i\n",
            dev_prop.maxThreadsPerBlock);
    printf("The maximum amount of threads per SM %i\n",
            dev_prop.maxThreadsPerMultiProcessor);
    printf("The maximum global memory is %zu\n", dev_prop.totalGlobalMem);
    printf("The total number of SM is %i\n", dev_prop.multiProcessorCount);
    printf("The number of register per block is %i\n", dev_prop.regsPerBlock);
    printf("The amount of shared memory per block %zu\n",
            dev_prop.sharedMemPerBlock);
    printf("The amount of shared memory per SM %zu\n",
            dev_prop.sharedMemPerMultiprocessor);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           dev_prop.memoryClockRate*(dev_prop.memoryBusWidth/8)/1.0e6);

}
