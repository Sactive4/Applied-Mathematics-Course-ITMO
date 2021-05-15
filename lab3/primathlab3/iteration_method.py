import numpy as np

# ПУНКТ 3
# Методы решения СЛАУ Зейделя
def seidel_method(A, b, eps):
    """Найти решение системы Ax=b методом Гаусса-Зейделя
    A - матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    returns:
    None - если решения не существует
    x - решение
    """
    n = A.shape[0]
    x = np.zeros(n)

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new


    print(x)
    return x

# test_solving_function(seidel_method)