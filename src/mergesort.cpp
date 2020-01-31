#include "../include/sedgewick.hpp"
#include <vector>
#include "../include/storage.h"

void GPUMergeSort::sort(std::vector<int>&a) {
    Storage data = Storage(a);
    Storage aux = Storage(std::vector<int>(a.size()));
    merge_sort(data.gpu_pointer(), data.gpu_pointer(), data.size());
}
