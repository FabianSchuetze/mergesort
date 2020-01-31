#include "../include/sedgewick.hpp"
#include <vector>
#include "../include/storage.h"
#include <iostream>
#include <sys/time.h>

double cpuSecond2() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
std::vector<int> GPUMergeSort::sort(std::vector<int>&a) {
    Storage data = Storage(a);
    Storage aux = Storage(std::vector<int>(a.size()));
    double gpuBegin = cpuSecond2();
    merge_sort(data.gpu_pointer(), aux.gpu_pointer(), data.size());
    double gpuEnd = cpuSecond2()- gpuBegin;
    std::cout << "the gpu took: " << gpuEnd << std::endl;
    std::vector<int> res = data.return_data();
    return res;
}
