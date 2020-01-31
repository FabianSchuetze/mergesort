#include <vector>
#include "../include/sedgewick.hpp"
#include <iostream>
#include <string>
#include <sys/time.h>

#include <random>

std::vector<int> variables(int size, int min, int max) {
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(min,max); // guaranteed unbiased
    std::vector<int> res(size);
    for (int i = 0; i < size; ++i) {
        res[i] = uni(rng);
    }
    return res;
}
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main() {
    //std::vector<int>a = {10, -5, 20, -100, -300};
    std::vector<int>a = variables(100000, -100000,100000);
    //for (auto i : a) {
        //std::cout << i << std::endl;
    //}
    GPUMergeSort seg;
    std::vector<int> result = seg.sort(a);
    //for (auto i : result) {
        //std::cout << i << ", ";
    //}
    //std::cout << "cpu\n" << std::endl;
    Sedgwick<int> seg_cpu;
    double cpuBegin = cpuSecond();
    seg_cpu.sort(a);;
    double cpuEnd = cpuSecond()- cpuBegin;
    std::cout << "the cpu took: " << cpuEnd << std::endl;
    //for (auto i : a) {
        //std::cout << i << ", ";
    //}
}
