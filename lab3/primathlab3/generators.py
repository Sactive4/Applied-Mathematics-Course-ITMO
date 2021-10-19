from .math_util import empty_matrix


def generate_diagonal_domination_matrix(a, k):
    """Генерирует тестовое уравнение для решения
    :param a: квадратная матрица коэффициентов a_ij (см. лабу, а также отчет)
    :param k: номер уравнения в последовательности
    :return: пара (A_k, F_k) для уравнения A_k * x_k = F_k
    None - если уравнение несовместно
    """
    n = len(a)
    A_k = empty_matrix(n, n, "lil")

    for i in range(n):
        t1 = -sum(a[i][k] for k in range(i))
        t2 = -sum(a[i][k] for k in range(i + 1, n))
        t = t1 + t2
        for j in range(n):
            if i != j:
                A_k[i, j] = a[i, j]
            else:
                A_k[i, j] = t + pow(10.0, -k)

    A_k = A_k.tocsr()
    return A_k


def generate_hilbert_matrix(k):
    """Генерирует тестовое уравнение для решения с матрицей Гильберта
    :param k: номер уравнения в последовательности
    :return: пара (A_k, F_k) для уравнения A_k * x_k = F_k
    None - если уравнение несовместно
    """
    A_k = empty_matrix(k, k, "lil")
    for i in range(k):
        for j in range(k):
            A_k[i, j] = 1.0 / (i + j + 1.0)

    return A_k.tocsr()