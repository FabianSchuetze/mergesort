#include <vector>
#include "../include/sedgewick.hpp"
#include <iostream>
#include <string>
int main() {
    //std::vector<std::string>a = {"m", "e", "r", "g", "e", "s", "o", "r", "t",
        //"e", "x", "a", "m", "p", "l", "e"};
    std::vector<int>a = {10, -5, 20, -100, -300};
    for (auto i : a) {
        std::cout << i << std::endl;
    }
    GPUMergeSort seg;
    seg.sort(a);
    for (auto i : a) {
        std::cout << i << ", ";
    }
}
