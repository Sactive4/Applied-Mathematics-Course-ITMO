import numpy as np

from .math_util import empty_matrix, identity_matrix


def lil_row_product(matrix1, matrix2, row_idx1, row_idx2):
    row1 = matrix1.rows[row_idx1]
    row2 = matrix2.rows[row_idx2]
    data1 = matrix1.data[row_idx1]
    data2 = matrix2.data[row_idx2]

    res = 0
    i1 = i2 = 0

    while i1 < len(row1) and i2 < len(row2):
        col_idx1 = row1[i1]
        col_idx2 = row2[i2]

        if col_idx1 == col_idx2:
            res += data1[i1] * data2[i2]
            i1 += 1
        elif col_idx1 < col_idx2:
            i1 += 1
        else:
            i2 += 1

    return res


def csr_row_iter(csr, row_idx):
    row_len = csr.shape[1]
    data_start = csr.indptr[row_idx]
    data_end = csr.indptr[row_idx + 1] if row_idx + 1 < csr.shape[0] else len(csr.data)

    j = 0

    for data_idx in range(data_start, data_end):
        col = csr.indices[data_idx]
        val = csr.data[data_idx]

        while j < col:
            yield 0
            j += 1
        
        yield val
        j += 1
    
    for _ in range(j, row_len):
        yield 0


def lu_decomposition(A):
    """Найти LU-разложение матрицы A
    A - матрица, хранящаяся в разреженном виде
    :returns
    (L, U) - пара матриц L и U в разреженном виде, где A = LU
    L - нижняя треугольная матрица
    U - верхняя треугольная матрица
    :throws
    A -
    а) квадратная
    б) невырожденная <=> detA != 0
    в) миноры не вырождены <=> every detAij != 0
    """
    # пример кода
    # https://en.wikipedia.org/wiki/LU_decomposition
    # конкретные формулы и реализация
    # https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/

    N = A.shape[0]
    L = identity_matrix(N, "lil")
    U = empty_matrix(N, N, "lil")  # транспонированная

    for i in range(N):
        for j, a in zip(range(N), csr_row_iter(A, i)):
            num = a - lil_row_product(U, L, j, i)

            if i <= j:
                U[j, i] = num
            else:
                L[i, j] = num / U[j, j]

    return L.tocsr(), U.transpose().tocsr()


def lower_trivial_system_solution(A, b):
    """Найти тривиальное решение уравнения Ax=B
    A - нижнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    x = np.zeros(len(b))
    x[0] = b[0]

    for i in range(1, len(b)):
        x[i] = b[i] - A[i] * x

    return x


def upper_trivial_system_solution(A, b):
    """Найти тривиальное решение уравнения Ax=B
    A - верхнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    N = len(b)
    x = np.zeros(N)
    x[-1] = b[-1] / A[-1, -1]

    for i in reversed(range(N - 1)):
        x[i] = (b[i] - A[i] * x) / A[i, i]

    return x


def system_solution(A, b):
    """Найти решение системы Ax=b
    A - невырожденная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    returns:
    None - если решения не существует
    """
    L, U = lu_decomposition(A)
    y = lower_trivial_system_solution(L, b)
    x = upper_trivial_system_solution(U, y)
    return x


### Обратите внимание!
# Совместность системы: https://ru.wikipedia.org/wiki/Теорема Кронекера — Капелли
