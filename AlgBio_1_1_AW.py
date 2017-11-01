# author: Alexander Werthmann
# Task Sheet 1 - Sequence alignment
# Task 1
import numpy as np


def score(ai, bj):
    if ai == '-' or bj == '-':
        return -4
    if ai == bj:
        return 3
    if ai != bj:
        return -2


def calc_matrix(a, b, d):
    """

    :param a: string a
    :param b: string b
    :param d:
    :return: the matrix for the needleman-wunsch algorithm
    """

    a = list(a)
    b = list(b)
    k = [0 for i in range(len(b)+1) for j in range(len(a)+1)]
    matrix = np.matrix(k).reshape(len(a)+1, len(b)+1)
    for i in range(0, matrix.shape[1]):
        matrix[0, i] = i*d
    for i in range(1, matrix.shape[0]):
        matrix[i, 0] = i*d
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            match = int(matrix[i - 1, j - 1]) + score(a[i-1], b[j-1])
            delete = int(matrix[i - 1, j]) + d
            insert = int(matrix[i, j - 1]) + d
            matrix[i, j] = max(match, delete, insert)
    print(matrix)
    return matrix


def get_alignment(a, b, d, matrix):
    alignment_a = ''
    alignment_b = ''
    i = len(a)
    j = len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i, j] == matrix[i-1, j-1] + score(a[i-1], b[j-1]):
            alignment_a = a[i-1] + alignment_a
            alignment_b = b[j-1] + alignment_b
            i = i-1
            j = j-1
        elif i > 0 and matrix[i, j] == (matrix[i-1, j] + d):
            alignment_a = a[i-1]+alignment_a
            alignment_b = '-'+alignment_b
            i = i-1
        else:
            alignment_a = '-'+alignment_a
            alignment_b = b[j-1]+alignment_b
            j = j-1
    print(alignment_a)
    print(alignment_b)
    return alignment_a, alignment_b


# Get alignments
a = 'TATAAT'
b = 'TTACGTAAGC'
d = -2
matrix = calc_matrix('TATAAT', 'TTACGTAAGC', d)
get_alignment(a, b, d, matrix)

# Solution
# Resulting alignment:
# TTACGTAAGC
# -TA--TAA-T
