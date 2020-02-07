#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
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
    std::map<int, double> gpu_times;
    std::map<int, double> cpu_times;
     std::vector<int> sizes = {500, 1000, 5000,10000, 20000, 50000};
    //std::vector<int> sizes = {50000};
    for (int size : sizes) {
        int width = 8000;
        vector<int> A = variables(size, width);
        vector<int> B = variables(size, 2 * width);
        vector<int> cpu(A.size() + B.size());
        //for (int i : A) std::cout << i << "\n";
        //std::cout << "\nAnd B contains:\n";
        //for (int i : B) std::cout << i << "\n";
        //std::cout << "\nmerged:\n";
        vector<int> C(A.size() + B.size());
        Storage s_a(A);
        Storage s_b(B);
        Storage s_c(C);
        gpu_times[size] =
            cuda_merge(s_a.gpu_pointer(), A.size(), s_b.gpu_pointer(),
                              B.size(), s_c.gpu_pointer(), 100);
        //for (int i : s_c.return_data_const()) {
            //std::cout << i << "\n";
        //}
        double begin = cpuSecond2();
        std::merge(A.begin(), A.end(), B.begin(), B.end(), cpu.begin());
        double end = cpuSecond2() - begin;
        cpu_times[size] = end;
        bool equal = std::equal(s_c.return_data_const().begin(),
                                s_c.return_data_const().end(), cpu.begin());
        if (!equal) {
            std::cout << "for size " << size << "the results are not equal\n";
        }
    }
    for (int i : sizes) {
        std::cout << i << ", " << cpu_times[i] << ", " << gpu_times[i]
                  << std::endl;
    }
}
