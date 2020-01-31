#include "../include/sedgewick.hpp"
#include <vector>
#include "../include/storage.h"
#include <iostream>

void GPUMergeSort::sort(std::vector<int>&a) {
    Storage data = Storage(a);
    Storage aux = Storage(std::vector<int>(a.size()));
    merge_sort(data.gpu_pointer(), aux.gpu_pointer(), data.size());
    std::vector<int> res2 = data.return_data();
    for (int i : res2){
        std::cout << i << ", ";
    }
    std::cout << "end2\n";
}
