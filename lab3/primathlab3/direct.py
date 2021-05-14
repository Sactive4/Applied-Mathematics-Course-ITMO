import numpy as np

from .math_util import empty_matrix, identity_matrix


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

    N = len(A.toarray())
    L = empty_matrix(N, N)
    U = empty_matrix(N, N)

    return L, U


# A = scipy.sparse.csr_matrix([[2, 0, 4], [1, 9, 0], [2, 0, 1]])
# lu_decomposition(A)


def lower_trivial_system_solution(A, b):
    """Найти тривиальное решение уравнения Ax=B
    A - нижнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    x = np.zeros(len(b))
    x[0] = b[0]

    for i in range(1, len(b)):
        row = A[i].toarray()[0]
        x[i] = b[i] - sum(row * x)
    return x


def upper_trivial_system_solution(A, b):
    """Найти тривиальное решение уравнения Ax=B
    A - верхнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    x = np.zeros(len(b))
    x[-1] = b[-1]

    for i in range(len(b) - 2, -1, -1):
        row = A[i].toarray()[0]
        x[i] = b[i] - sum(row * x)
    return x


def system_solution(A, b):
    """Найти решение системы Ax=b
    A - матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    returns:
    None - если решения не существует
    """
    L, U = lu_decomposition(A)
    y = lower_trivial_system_solution(L, b)
    x = upper_trivial_system_solution(U, y)
    return x


def system_solution_matrix(A, B):
    """Найти решение системы AX=B
    A - матрица, хранящаяся в разреженном виде
    B - матрица, хранящаяся в разреженном виде
    returns:
    X - решение-матрица в разреженном виде
    """
    X = []
    for i in range(B.количество_столбцов):
        # i - индекс столбца
        X[i] = system_solution(A, B[i])
    return X


def inverse_matrix(A):
    """Найти обратную для матрицы A
    A - матрица, хранящаяся в разреженном виде
    :returns
    A_inverse - обратная матрица в разреженном виде
    """
    return system_solution_matrix(A, identity_matrix(A.размерность))


### Обратите внимание!
# Совместность системы: https://ru.wikipedia.org/wiki/Теорема Кронекера — Капелли
