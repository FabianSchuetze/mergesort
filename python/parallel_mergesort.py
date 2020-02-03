r"""
Tries to parallelize merge
"""


def mergepath(list_a, count_a, list_b, count_b, diag):
    """Tries to find the mergepath"""
    import pdb; pdb.set_trace()
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
          list_c, start_c, length):
    """Sequential merge"""
    import pdb; pdb.set_trace()
    i = 0
    k = 0
    # offset = n_diag * count_diag
    j = 0
    while (k < length):
        if (start_a == count_a):
            list_c[start_c + k] = list_b[start_b + j]
            j += 1
        elif (start_b == count_b):
            list_c[start_c + k] = list_a[start_a + i]
            i += 1
        elif (list_a[start_a + i] < list_b[start_b + j]):
            list_c[start_c + k] = list_a[start_a + i]
            i += 1
        else:
            list_c[start_c + k] = list_b[start_b + j]
            j += 1
        k += 1


def parallelMerge(process, list_a, list_b, list_c, length):
    """
    Tries to do the paralel merge for process p"""
    diag = process * length
    a_start, b_start = mergepath(list_a, len(list_a), list_b, len(list_b),
                                 diag)
    merge(list_a, a_start, len(list_a), list_b, b_start, len(list_b),
          list_c, diag, length)


if __name__ == "__main__":
    A = [17, 29, 35, 73, 86, 90, 95, 99]
    C = [0]*16
    B = [3, 5, 12, 22, 45, 64, 69, 82]
    PREV_CUTOFF = 0
    PREV_INVERSE = 0
    IT = 0
    for process in range(4):
        parallelMerge(process, A, B, C, 4)
