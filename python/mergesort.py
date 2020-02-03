r"""
Tries to write the merge path problem as a sequential problem"""


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
    A = [17, 29, 35, 73, 86, 90, 95, 99]
    C = [0]*16
    B = [3, 5, 12, 22, 45, 64, 69, 82]
    PREV_CUTOFF = 0
    PREV_INVERSE = 0
    IT = 0
    for DIAG in [4, 8, 12, 16]:
        CUTOFF = mergepath(A, 8, B, 8, DIAG)
        INV = DIAG - CUTOFF
        print("For diag %i, the accpected list A is:" % (DIAG), end='')
        print(A[PREV_CUTOFF:CUTOFF], end=' ')
        print('and the list for B is: ', end='')
        print(B[PREV_INVERSE: INV])
        merge(A[PREV_CUTOFF:CUTOFF], CUTOFF - PREV_CUTOFF,
              B[PREV_INVERSE:INV], C, 4, IT)
        PREV_CUTOFF = CUTOFF
        PREV_INVERSE = INV
        IT += 1
