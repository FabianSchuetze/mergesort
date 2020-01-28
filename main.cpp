#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "functions.h"
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
    int x_prev(-1000000);
    while (is >> x) {
        if (x < x_prev) {
            throw std::runtime_error("Arrays must be ordered");
        }
        array.push_back(x);
        x_prev = x;
    }
    return array;
}
int main() {
    vector<int> A = read_array("A.out");
    vector<int> B = read_array("B.out");
    vector<int> C;
    vector<int> D;
    C.resize(A.size() + B.size());
    D.resize(A.size() + B.size());
    //for (int i : A) {
        //std::cout << i << ", ";
    //}
    //std::cout << "\nB\n";
    //for (int i : B) {
        //std::cout << i << ", ";
    //}
    std::cout << "\n The size of C is: " << C.size() << std::endl;
    int * p_c = C.data();
    int * p_d = D.data();
    double cpuStart = cpuSecond2();
    calling_function(A.data(), A.size(),  B.data(), B.size(), p_c);
    double cpuEnd = cpuSecond2() - cpuStart;
    std::cout << "sequential took: " << cpuEnd << std::endl;
    std::ofstream outFile1("C_out.txt");
    // the important part
    for (const auto &c : C) outFile1 << c << "\n";
    //for (int c : C) {
        //std::cout << c << std::endl;
    //}
    merge(A.data(), A.size(),  B.data(), B.size(), p_d);
    std::cout << "D array" << std::endl;
    std::ofstream outFile("D_out.txt");
    // the important part
    for (const auto &d : D) outFile << d << "\n";
    //for (int d : D) {
        //std::cout << d << std::endl;
    //}
}
