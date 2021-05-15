from .direct import system_solution
from .math_util import empty_matrix, identity_matrix


def system_solution_matrix(A, B):
    """Найти решение системы AX=B
    A - матрица, хранящаяся в разреженном виде
    B - матрица, хранящаяся в разреженном виде
    returns:
    X - решение-матрица в разреженном виде
    """
    N = A.shape[0]
    X = empty_matrix(N, N).tolil()
    for i in range(N):
        X[i] = system_solution(A, B.getcol(i).toarray())
    return X.tocsr().transpose()


def inverse_matrix(A):
    """Найти обратную для матрицы A
    A - матрица, хранящаяся в разреженном виде
    :returns
    A_inverse - обратная матрица в разреженном виде
    """
    return system_solution_matrix(A, identity_matrix(A.shape[0]))
