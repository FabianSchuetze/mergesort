r"""
Tries to parallelize merge
"""
import time
import numpy as np
import multiprocessing
from multiprocessing import Pool


def mergepath(list_a, count_a, list_b, count_b, diag):
    """Tries to find the mergepath"""
    # import pdb; pdb.set_trace()
    if diag == 0:
        return 0, 0
    begin = max(0, diag - count_b)
    end = min(diag, count_a)

    while begin < end:
        mid = (begin + end) >> 1
        a_key = list_a[mid]
        b_key = list_b[diag - 1 - mid]
        pred = a_key < b_key
        if pred:
            begin = mid + 1
        else:
            end = mid
    b_begin = diag - begin
    return begin, b_begin

def merge(list_a, start_a, count_a, list_b, start_b, count_b,
          start_c, length):
    """Sequential merge"""
    # import pdb; pdb.set_trace()
    i = 0
    k = 0
    j = 0
    while k < length:
        if start_a + i == count_a:
            C[start_c + k] = list_b[start_b + j]
            j += 1
        elif start_b + j == count_b:
            C[start_c + k] = list_a[start_a + i]
            i += 1
        elif list_a[start_a + i] <= list_b[start_b + j]:
            C[start_c + k] = list_a[start_a + i]
            i += 1
        else:
            C[start_c + k] = list_b[start_b + j]
            j += 1
        k += 1
    # sleep = np.random.rand()
    # time.sleep(sleep)
    # print(list_c)


def parallelMerge(process, list_a, list_b, length):
    """
    Tries to do the paralel merge for process p"""
    diag = process * length
    a_start, b_start = mergepath(list_a, len(list_a), list_b, len(list_b),
                                 diag)
    merge(list_a, a_start, len(list_a), list_b, b_start, len(list_b),
          diag, length)


if __name__ == "__main__":
    SIZE = int(4 ** 10)
    A = list(np.sort(np.random.randint(-10000, 10000, int(SIZE / 2))))
    B = list(np.sort(np.random.randint(-10000, 10000, int(SIZE / 2))))
    # A = [17, 29, 35, 73, 86, 90, 95, 99]
    C = [0]*SIZE
    # C = multiprocessing.Array('i', C1, lock=False)
    # B = [3, 5, 12, 22, 45, 64, 69, 82]
    inputs = [(i, A, B, int(SIZE / 4)) for i in range(4)]
    start = time.time()
    # for step in range(4):
        # parallelMerge(step, A, B, int(SIZE / 4))
    with Pool(4) as pool:
        pool.starmap(parallelMerge, inputs)
    end = time.time()
    print("It took %.5f second\n" %(end - start))

