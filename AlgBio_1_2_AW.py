

# author Alexander Werthmann
# Task 2 - Shortest common superstring

import numpy as np

def super_string(a, b):
    m = len(a)
    n = len(b)
    k = [0 for i in range(m+1) for j in range(n+1)]
    matrix = np.matrix(k).reshape(m + 1, n + 1)
    scs = ''

    # Step 1 create matrix
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                matrix[i, j] = j
            elif j == 0:
                matrix[i, j] = i
            elif a[i-1] == b[j-1]:
                matrix[i, j] = matrix[i-1, j-1] + 1
            else:
                # find best predecessor
                matrix[i, j] = min(matrix[i-1, j], matrix[i, j-1]) + 1
    print(matrix)
    # Step 2 Iterate beginning from bottom right corner
    i = m
    j = n
    while i > 0 and j > 0:
        # same suffix
        if a[i-1] == b[j-1]:
            scs = a[i-1] + scs
            i = i - 1
            j = j - 1
        # not same suffix, find larger predecessor
        elif matrix[i-1, j] < matrix[i, j-1]:
            scs = a[i-1] + scs
            i = i - 1
        else:
            scs = b[j-1] + scs
            j = j - 1

    # Step 3 Append remaining chars of the longer string (if len(a) != len (b))
    while i > 0:
        scs = a[i-1] + scs
        i = i - 1
    while j > 0:
        scs = b[j-1] + scs
        j = j - 1
    return scs


scs = super_string("gnomes", "home")
print(scs)  # gnhomes

# Runtime evaluation:
# Given strings a and b
# m = len(a)
# n = len(b)
# 1) The matrix creation in step 1 takes O(m*n) time
# 2) The iterating process in step 2 takes at most O(m+n) time
# 3) The loops in the final step 3 takes max(m, n)-min(m, n) time (Only one loop will be accessed)
# This leads to a total amount of O(m*n) for the algorithm
