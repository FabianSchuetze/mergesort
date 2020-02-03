#include "../include/sedgewick.hpp"
#include <iostream>

template <class T>
void Sedgwick<T>::merge(T* a, int lo, int mid, int hi, T* tmp) {
    //std::cout << "merge(" << lo << ", " << mid << ", " << hi << ")\n";
    int i = lo;
    int j = mid + 1;

    for (int k = lo; k <= hi; k++) tmp[k] = a[k];

    for (int k = lo; k <= hi; k++) {
        if (i > mid)
            a[k] = tmp[j++];
        else if (j > hi)
            a[k] = tmp[i++];
        else if (tmp[j] < tmp[i])
            a[k] = tmp[j++];
        else
            a[k] = tmp[i++];
    }
}

template <typename T>
void Sedgwick<T>::sort(T* a, int lo, int hi, T* tmp) {
    if (hi <= lo) return;
    //std::cout << "sort(" << lo << ", " << hi << ")\n";
    int mid = lo + (hi - lo) / 2;
    sort(a, lo, mid, tmp);
    sort(a, mid + 1, hi, tmp);
    merge(a, lo, mid, hi, tmp);
}

template <class T>
void Sedgwick<T>::sort(std::vector<T>& a) {
    std::vector<T> aux(a.size());
    ////aux.resize(a.size());
    sort(a.data(), 0, a.size() - 1, aux.data());
}
template class Sedgwick<int>;
template class Sedgwick<std::string>;
