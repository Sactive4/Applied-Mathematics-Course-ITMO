# реализации разреженных матриц и введение
# https://www.machinelearningmastery.ru/sparse-matrices-for-machine-learning/
import time

import numpy
import scipy
import scipy.sparse

numpy.seterr(all='raise')
import warnings
warnings.simplefilter('error')

from primathlab3.math_util import (
    ascending_vector,
    generate_big_matrix,
    random_vector, equals,
)

# ПУНКТ 1
# LU-композиция
# решение системы уравнений (LU + Гаусса)
# обратная матрица

# См. direct.py

# ПУНКТ 2
# Тестирование программы из п. 1

# См. tests/test_direct.py

#
# # ПУНКТ 3
# # Методы решения СЛАУ Сейделя
# # P.S. в чате написано, что достаточно только одного метода
#
#
# def seidel_method(A, b):
#     """Найти решение системы Ax=b методом Гаусса-Зейделя
#     A - матрица, хранящаяся в разреженном виде
#     b - вектор в правой части
#     returns:
#     None - если решения не существует
#     x - решение
#     """
#
#     return x
#
#
# test_solving_function(seidel_method)
#
#
# # ПУНКТ 4
# # насколько я понимаю, используется метод из п. 3
# # оценка влияния увеличения числа обусловленности на точность решения
#
#
# def generate_test_equation(a, k):
#     """Генерирует тестовое уравнение для решения
#     :param A: квадратная матрица коэффициентов a_ij (см. лабу, а также чат - опечатка в лабе)
#     :param k: номер уравнения в последовательности
#     :return: пара (A_k, F_k) для уравнения A_k * x_k = F_k
#     None - если уравнение несовместно
#     """
#     A_k = []  # генерируется матрица по соотношению из лабы
#     F_k = A_k * ascending_vector(A.размерность)
#     if несовместна(A_k):
#         return None
#     return (A_k, F_k)
#
#
# def test_equations(fn, n):
#     """Возвращает массив погрешностей для последовательности
#     :param fn: тестируемая функция генерации уравнений lambda k
#     :param n: количество уравнений последовательности
#     :return: массив размера n с погрешностью = max(x_1, x_2, ..., x_n)
#     где x_i - погрешность между точным решением и решением метода ??? (наверное, так)
#     """
#     r = []
#     for i in range(n):
#         x = fn(i)
#         r_i = 0.0
#         for j in range(len(x)):
#             r_i = max(r_i, x[j])
#         r.append(r_i)
#     return r
#
#
# a = [[-2, -1, 0], [-7, -1, -9], [-3, -11, -2]]  # например
# r = test_equations(lambda k: generate_test_equation(a, k), 20)
# # todo: построить график для последовательности и обработать результат
#
#
# # ПУНКТ 5
# # исследования на матрицах Гильберта
#
#
# def generate_test_equation_hilbert(k):
#     """Генерирует тестовое уравнение для решения с матрицей Гильберта
#     :param k: номер уравнения в последовательности
#     :return: пара (A_k, F_k) для уравнения A_k * x_k = F_k
#     None - если уравнение несовместно
#     """
#     A_k = []  # генерируется матрица по соотношению из лабы для гилберта
#     F_k = A_k * ascending_vector(A.размерность)
#     if несовместна(A_k):
#         return None
#     return (A_k, F_k)
#
#
# r = test_equations(generate_test_equation_hilbert, 20)
# # todo: построить график для последовательности и обработать результат
#
#
# # ПУНКТ 6
# # сравнение прямого и итерационного метода
# # прямой - system_solution(A, b)
# # итерационный - seidel_method(A, b)
#
#
# def method_comparison(p):
#     """Сравнить прямой и итерационный метод на матрицах
#     разреженности p
#     :param p: разреженность, т.е. отношение ненулевых клеток к nxn
#     :return: [(t1, t2)] - массив пар с временем (у.е.) решения уравнений двумя методами
#     """
#     ns = [10, 50, 100, 1000, 10000, 100000, 1000000]
#     r = []
#     for i in range(len(ns)):
#         A = generate_big_matrix(ns[i], p)
#         b = random_vector(ns[i])
#
#         t0 = time.time()
#         system_solution(A, b)
#         t1 = time.time() - t0
#         seidel_method(A, b)
#         t2 = time.time() - t1
#
#         r.append((t1, t2))
#     return r
#
#
# # todo: построить график для последовательности и обработать результат
#

# ПУНКТ 7
# реализация поиска обратной матрицы

# См. inverse.py, test_inverse.py

