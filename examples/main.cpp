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

// void dump_files(const vector<int>& gpu) {
// std::ofstream outFile1("check_gpu_out.txt");
// for (const auto& c : gpu) outFile1 << c << "\n";
//}

int main() {
    int size = 8;
    int width = 20;
    vector<int> A = variables(size, width);
    vector<int> B = variables(size, width);
    for (int i : A) 
        std::cout << i << ", ";
    std::cout << "\nAnd B contains:\n";
    for (int i : B)
        std::cout << i << ", ";
    std::cout << "\nmerged:\n";
    vector<int> C(A.size() + B.size());
    Storage s_a(A);
    Storage s_b(B);
    Storage s_c(C);
    cuda_merge(s_a.gpu_pointer_const(), A.size(), s_b.gpu_pointer_const(),
               B.size(), s_c.gpu_pointer(), 8);
    for (int i : s_c.return_data_const()) {
        std::cout << i << ", ";
    }
    // dump_files(C);
}

