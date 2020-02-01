#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "../include/merge.h"
#include <sys/time.h>
#include <random>
#include <algorithm>

using std::vector;

double cpuSecond2() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
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
//vector<int> read_array(const std::string& file) {
    //std::vector<int> array;
    //std::ifstream is(file);
    //int x;
    //int x_prev(-1000000); //curde way of saying -INT32
    //while (is >> x) {
        //if (x < x_prev) {
            //throw std::runtime_error("Arrays must be ordered");
        //}
        //array.push_back(x);
        //x_prev = x;
    //}
    //return array;
//}
void dump_files(const vector<int>& gpu) {
    std::ofstream outFile1("check_gpu_out.txt");
    for (const auto &c : gpu) outFile1 << c << "\n";
}

int main() {
    int size = std::pow(2, 15);
    vector<int> A;
    for (int i = 0; i < 4; ++i) {
        vector<int> tmp = variables(size, -100000, 100000);
        std::sort(tmp.begin(), tmp.end());
        A.insert(A.end(), tmp.begin(), tmp.end());
    }
    //vector<int> A = read_array("A.out");
    //vector<int> B = read_array("B.out");
    //int A_size = A.size();
    //A.insert( A.end(), B.begin(), B.end() );
    vector<int> C(A.size());
    //vector<int> D(A.size() + B.size());
    cuda_merge(A.data(), size*2,  size*2, C.data());
    double cpuStart = cpuSecond2();
    std::sort(A.begin(), A.end());
    double end = cpuSecond2() - cpuStart;
    std::cout << "The cpu took: " << end << std::endl;
    dump_files(C);
}
