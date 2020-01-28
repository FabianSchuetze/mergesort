#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "../include/merge.h"
#include <sys/time.h>

using std::vector;

double cpuSecond2() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
vector<int> read_array(const std::string& file) {
    std::vector<int> array;
    std::ifstream is(file);
    int x;
    int x_prev(-1000000); //curde way of saying -INT32
    while (is >> x) {
        if (x < x_prev) {
            throw std::runtime_error("Arrays must be ordered");
        }
        array.push_back(x);
        x_prev = x;
    }
    return array;
}
void dump_files(const vector<int>& cpu, const vector<int>& gpu) {
    std::ofstream outFile1("gpu_out.txt");
    for (const auto &c : gpu) outFile1 << c << "\n";
    std::ofstream outFile("cpu_out.txt");
    for (const auto &d : cpu) outFile << d << "\n";
}
int main() {
    vector<int> A = read_array("A.out");
    vector<int> B = read_array("B.out");
    vector<int> C(A.size() + B.size());
    vector<int> D(A.size() + B.size());
    cuda_merge(A.data(), A.size(),  B.data(), B.size(), C.data());
    double cpuStart = cpuSecond2();
    merge(A.data(), A.size(),  B.data(), B.size(), D.data());
    double cpuEnd = cpuSecond2() - cpuStart;
    std::cout << "sequential took: " << cpuEnd << std::endl;
    dump_files(C, D);
}
