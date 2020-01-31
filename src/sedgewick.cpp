#include "../include/sedgewick.hpp"
#include <iostream>

template <class T>
void Sedgwick<T>::merge(std::vector<T>& a, int lo, int mid, int hi) {
    //std::cout << "merge(" << lo << ", " << mid << ", " << hi << ")\n";
    int i = lo;
    int j = mid + 1;

    for (int k = lo; k <= hi; k++) aux[k] = a[k];

    for (int k = lo; k <= hi; k++) {
        if (i > mid)
            a[k] = aux[j++];
        else if (j > hi)
            a[k] = aux[i++];
        else if (aux[j] < aux[i])
            a[k] = aux[j++];
        else
            a[k] = aux[i++];
    }
}

template <typename T>
void Sedgwick<T>::sort(std::vector<T>& a, int lo, int hi) {
    if (hi <= lo) return;
    //std::cout << "sort(" << lo << ", " << hi << ")\n";
    int mid = lo + (hi - lo) / 2;
    sort(a, lo, mid);
    sort(a, mid + 1, hi);
    merge(a, lo, mid, hi);
}

template <class T>
void Sedgwick<T>::sort(std::vector<T>& a) {
    aux.resize(a.size());
    sort(a, 0, a.size() - 1);
}
template class Sedgwick<int>;
template class Sedgwick<std::string>;
