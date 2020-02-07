#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "../include/merge.h"
#include "../include/storage.h"

using std::vector;

double cpuSecond2() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
std::vector<int> variables(int size, int width) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(-width, width);
    std::vector<int> res(size);
    for (int i = 0; i < size; ++i) {
        res[i] = uni(rng);
    }
    std::sort(res.begin(), res.end());
    return res;
}

int main() {
    int size = 50000;
    int width = 8000;
    vector<int> A = variables(size, width);
    vector<int> B = variables(size, 2 * width);
    vector<int> cpu(A.size() + B.size());
    // for (int i : A) std::cout << i << "\n";
    // std::cout << "\nAnd B contains:\n";
    // for (int i : B) std::cout << i << "\n";
    // std::cout << "\nmerged:\n";
    vector<int> C(A.size() + B.size());
    Storage s_a(A);
    Storage s_b(B);
    Storage s_c(C);
    cuda_merge(s_a.gpu_pointer(), A.size(), s_b.gpu_pointer(), B.size(),
               s_c.gpu_pointer(), 8);
    // for (int i : s_c.return_data_const()) {
    // std::cout << i << "\n";
    //}
    double begin = cpuSecond2();
    std::merge(A.begin(), A.end(), B.begin(), B.end(), cpu.begin());
    double end = cpuSecond2() - begin;
    bool equal = std::equal(s_c.return_data_const().begin(),
                            s_c.return_data_const().end(), cpu.begin());
    std::cout << "the two are equal: " << equal << std::endl;
    std::cout << "It took the CPU: " << end << std::endl;
}
