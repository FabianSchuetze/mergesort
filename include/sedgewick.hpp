#ifndef Sedgwick_hpp
#define Sedgwick_hpp
#include <vector>
template <class T>
class Sedgwick {
   public:
    void sort(std::vector<T>&);

   private:
    std::vector<T> aux;
    void sort(std::vector<T>& a, int low, int high);
    void merge(std::vector<T>& a, int low, int mid, int high);
};

class GPUMergeSort {
   public:
   std::vector<int> sort(std::vector<int>&);

   private:
    std::vector<int> aux;
    void sort(std::vector<int>& a, int low, int high);
    void merge(std::vector<int>& a, int low, int mid, int high);
};


void merge_sort(int* src, int* dest, int size);
#endif
