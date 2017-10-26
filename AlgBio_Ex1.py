import numpy as np


def score(ai, bj):
    if ai == '-' or bj =='-':
        return -4
    if ai == bj:
        return 3
    if ai != bj:
        return -2



def calc_matrix(a, b, d):
    a = list(a)
    b = list(b)
    k = ["1abc" for i in range(len(b)+1) for j in range(len(a)+1)]
    matrix = np.matrix(k).reshape(len(a)+1, len(b)+1)
    for i in range(0, matrix.shape[1]):
        matrix[0, i] = str(i*d)
    for i in range(1, matrix.shape[0]):
        matrix[i, 0] = str(i*d)
    print(matrix)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            print(str(i)+" "+str(j)+"  shape "+str(matrix.shape[0])+" "+str(matrix.shape[1]))
            match = str(int(matrix[i - 1, j - 1]) + score(a[i-1], b[j-1]))
            delete = str(int(matrix[i - 1, j]) + d)
            insert = str(int(matrix[i, j - 1]) + d)
            matrix[i,j] = str(max(match,delete,insert))
    print(matrix)
    return matrix


calc_matrix('TATAAT' , 'TTACGTAAGC', -2)
