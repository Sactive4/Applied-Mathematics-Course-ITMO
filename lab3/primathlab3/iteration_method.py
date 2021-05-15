import math
import warnings

import numpy as np
import scipy
from numpy import array, zeros, diag, diagflat, dot


# ПУНКТ 3
# Методы решения СЛАУ мтодом Якоби

def norm(x):
    return math.sqrt(np.dot(x, x))


def jacobi(matrix, b, x, tolerance):
    matrix = scipy.sparse.csc_matrix(matrix)

    ITERATION_LIMIT = 10000
    ERROR_LIMIT = 1e100

    for it_count in range(ITERATION_LIMIT):
        x_new = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            s1 = np.dot(matrix.toarray()[i, :i], x[:i])
            s2 = np.dot(matrix.toarray()[i, (i + 1):], x[(i + 1):])
            x_new[i] = (1.0 * b[i] - s1 - s2) / matrix[i, i]
        error = norm(np.dot(matrix.toarray(), x) - b)
        if error < tolerance or error > ERROR_LIMIT:
            break
        x = x_new

    print(x)
    return x
