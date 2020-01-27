#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "functions.h"

using std::vector;

vector<int> read_array(const std::string& file) {
    std::vector<int> array;
    std::ifstream is(file);
    int x;
    int x_prev(-100000);
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
    vector<int> A = read_array("a.txt");
    vector<int> B = read_array("b.txt");
    vector<int> C;
    C.resize(A.size() + B.size());
    for (int i : A) {
        std::cout << i << ", ";
    }
    std::cout << "\nB\n";
    for (int i : B) {
        std::cout << i << ", ";
    }
    std::cout << "\n The size of C is: " << C.size() << std::endl;
    int * p_c = C.data();
    calling_function(A.data(), A.size(),  B.data(), B.size(), p_c);
    for (int c : C) {
        std::cout << c << std::endl;
    }
}
