from itertools import count
from timeit import timeit

import numpy
from scipy.sparse import linalg

from primathlab3 import direct, iteration_method
from primathlab3.math_util import generate_big_matrix, random_vector

from lab3.primathlab3.math_util import empty_matrix


def gen_nonsingular_matrix(n, p):
    matrix = generate_big_matrix(n, p, "lil")
    for i in range(n):
        matrix[i, i] += 1000
    return matrix

def generate_diagonal_domination_matrix(a, k):
    """Генерирует тестовое уравнение для решения
    :param a: квадратная матрица коэффициентов a_ij (см. лабу, а также отчет)
    :param k: номер уравнения в последовательности
    :return: пара (A_k, F_k) для уравнения A_k * x_k = F_k
    None - если уравнение несовместно
    """
    n = len(a)
    A_k = empty_matrix(n, n, "lil").tolil()

    for i in range(n):
        t1 = -sum(a[i][k] for k in range(i))
        t2 = -sum(a[i][k] for k in range(i + 1, n))
        t = t1 + t2
        for j in range(n):
            if i != j:
                A_k[i, j] = t
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
    A_k = empty_matrix(k, k, "lil").tolil()
    for i in range(k):
        for j in range(k):
            A_k[i, j] = 1.0 / (i + j - 1.0)

    return A_k.tocsr()

def test_equations(A, F):
    """Возвращает сумму погрешностей для последовательности
    :param fn: тестируемая функция генерации уравнений lambda k
    :param n: количество уравнений последовательности
    :return: массив размера n с погрешностью = max(x_1, x_2, ..., x_n)
    где x_i - погрешность между точным решением и решением метода ???
    """

    sum = 0.0
    left = iteration_method.seidel_method(A, F)
    right = direct.system_solution(A, F)

    for i in range(left.shape[0]):
        sum = abs(right[i] - left[i])

    return sum

def gen_test_data(n, p=0.3):
    # Различные способы генерации матриц
    matrix = gen_nonsingular_matrix(n, p).tocsr()
    x = random_vector(n)
    vector = matrix * x
    return matrix, vector


def test_exe_time(solver, matrix, vector, repeat=1):
    return timeit(lambda: solver(matrix, vector), number=repeat)


def scipy_solver(matrix, vector):
    lu = linalg.splu(matrix)
    return lu.solve(vector)


if __name__ == "__main__":
    solver = iteration_method.seidel_method

    print("n, execution_time")

    for exp in count(1):
        n = 10 + exp
        matrix, vector = gen_test_data(n, p=0.01)
        exe_time = test_exe_time(solver, matrix, vector)
        print(n, exe_time, sep=", ")
