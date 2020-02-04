r"""
Tries to write the merge path problem as a sequential problem"""
import numpy as np

# def intersection(list_a, count_a, list_b, diag):
    # import pdb; pdb.set_trace()
    # a_top = min(diag, count_a)
    # b_top = max(diag - count_a, 0)
    # a_bottom = b_top

    # while True:
        # mid = (a_top - a_bottom) >> 1
        # a_key = a_top - mid
        # b_key = b_top + mid
        # val_b = list_b[b_key - 1] if b_key > 0 else -1000000
        # if list_a[a_key] > val_b:
            # val_a = list_a[a_key - 1] if a_key > 0 else - 1000000
            # if val_a <= list_b[b_key]:
                # a_start = a_key
                # b_start = b_key
                # break
            # else:
                # a_top = a_key - 1
                # b_top = b_key + 1
        # else:
            # a_bottom = a_key + 1
    # return a_start, b_start

def mergepath(list_a, count_a, list_b, count_b, diag):
    """Tries to find the mergepath"""
    # import pdb; pdb.set_trace()
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
    return begin


def merge(list_a, count_a, list_b, list_c, n_diag, count_diag):
    """Sequential merge"""
    i = 0
    k = 0
    offset = n_diag * count_diag
    j = 0
    while (i < count_a and j < (n_diag - count_a)):
        if list_a[i] < list_b[j]:
            list_c[offset + k] = list_a[i]
            i += 1
        else:
            list_c[offset + k] = list_b[j]
            j += 1
        k += 1
    if i == count_a:
        while j < n_diag - count_a:
            list_c[offset + k] = list_b[j]
            k += 1
            j += 1
    else:
        while i < count_a:
            list_c[offset + k] = list_a[i]
            k += 1
            i += 1


if __name__ == "__main__":
    # A = [17, 29, 35, 73, 86, 90, 95, 99]
    A = list(np.sort(np.random.randint(-100, 100, 32)))
    B = list(np.sort(np.random.randint(-100, 100, 32)))
    C = [0]*64
    # B = [3, 5, 12, 22, 45, 64, 69, 82]
    PREV_CUTOFF = 0
    PREV_INVERSE = 0
    IT = 0
    SIZE = 16
    for DIAG in [16, 32, 48, 64]:
        CUTOFF = mergepath(A, 32, B, 32, DIAG)
        # a_s, b_s = intersection(A, 8, B, DIAG)
        INV = DIAG - CUTOFF
        print("Diag %i, list A is:" % (DIAG), end='')
        print(A[PREV_CUTOFF:CUTOFF], end=' ')
        print("ending at (%i, %i) " % (CUTOFF, INV), end=' ')
        print('and the list for B is: ', end='')
        print(B[PREV_INVERSE: INV])
        merge(A[PREV_CUTOFF:CUTOFF], CUTOFF - PREV_CUTOFF,
              B[PREV_INVERSE:INV], C, 16, IT)
        PREV_CUTOFF = CUTOFF
        PREV_INVERSE = INV
        IT += 1
