import numpy
import numpy as np
import scipy.linalg

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
    L = identity_matrix(N)
    U = empty_matrix(N, N)

    for i in range(N):
        for j in range(N):
            if i <= j:
                U[i, j] = A[i, j] - sum(U[k, j] * L[i, k] for k in range(i))
            else:
                L[i, j] = 1.0 / U[j, j] * (A[i, j] - sum(U[k, j] * L[i, k] for k in range(j)))

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
        # todo: оптимизировать суммы
        x[i] = b[i] - sum(A[i, j] * x[j] for j in range(i))
        #row = A[i].toarray()[0]
        #x[i] = b[i] - sum(row * x)
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
        # todo: оптимизировать суммы
        x[i] = 1.0 / A[i, i] * (b[i] - sum(A[i, j] * x[j] for j in range(i + 1, N)))

    # x = np.zeros(len(b))
    # x[-1] = b[-1]
    #
    # for i in range(len(b) - 2, -1, -1):
    #     row = A[i].toarray()[0]
    #     x[i] = b[i] - sum(row * x)
    return x


def system_solution(A, b):
    """Найти решение системы Ax=b
    A - матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    returns:
    None - если решения не существует
    """
    det = numpy.linalg.det(A.todense())
    if det == 0.0:
        return None

    L, U = lu_decomposition(A)
    # A_l = A.toarray()
    # L_l = L.toarray()
    # U_l = U.toarray()
    #P_r, L_r, U_r = scipy.linalg.lu(A.toarray())
    y = lower_trivial_system_solution(L, b)
    x = upper_trivial_system_solution(U, y)
    return x


# def system_solution_matrix(A, B):
#     """Найти решение системы AX=B
#     A - матрица, хранящаяся в разреженном виде
#     B - матрица, хранящаяся в разреженном виде
#     returns:
#     X - решение-матрица в разреженном виде
#     """
#     X = []
#     for i in range(B.количество_столбцов):
#         # i - индекс столбца
#         X[i] = system_solution(A, B[i])
#     return X
#
#
# def inverse_matrix(A):
#     """Найти обратную для матрицы A
#     A - матрица, хранящаяся в разреженном виде
#     :returns
#     A_inverse - обратная матрица в разреженном виде
#     """
#     return system_solution_matrix(A, identity_matrix(A.размерность))


### Обратите внимание!
# Совместность системы: https://ru.wikipedia.org/wiki/Теорема Кронекера — Капелли
