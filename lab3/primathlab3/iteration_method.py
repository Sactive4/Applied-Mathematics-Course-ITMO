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
        x_new = x

        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new


    print(x)
    return x



# def seidel_method(A, b, eps):
#     """Найти решение системы Ax=b методом Гаусса-Зейделя
#     A - матрица, хранящаяся в разреженном виде
#     b - вектор в правой части
#     returns:
#     None - если решения не существует
#     x - решение
#     :type A: object
#     """
#     n = A.shape[0]
#     x = b
#     x_new = x
#
#     converge = False
#     while not converge:
#         # x_new = np.copy(x)
#
#         for j in range(0, n):
#             # temp variable d to store b[j]
#             d = b[j]
#
#             # to calculate respective xi, yi, zi
#             for i in range(0, n):
#                 if (j != i):
#                     d -= A[j, i] * x[i]
#             # updating the value of our solution
#             x_new[j] = d / A[j, j]
#
#         #
#         # for i in range(n):
#         #     s1 = sum(A[i, j] * x_new[j] for j in range(i))
#         #     s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
#         #     x_new[i] = (b[i] - s1 - s2) / A[i, i]
#         #
#         converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
#         x = x_new
#         # x = x_new
#
#
#     print(x)
#     return x
#
#
# # n = len(a)
# # # for loop for 3 times as to calculate x, y , z
# # for j in range(0, n):
# #     # temp variable d to store b[j]
# #     d = b[j]
# #
# #     # to calculate respective xi, yi, zi
# #     for i in range(0, n):
# #         if (j != i):
# #             d -= a[j][i] * x[i]
# #     # updating the value of our solution
# #     x[j] = d / a[j][j]
# # # returning our updated solution
# # return x
# #
# # # int(input())input as number of variable to be solved
# # n = 3
# # a = []
# # b = []
# # # initial solution depending on n(here n=3)
# # x = [0, 0, 0]
# # a = [[4, 1, 2], [3, 5, 1], [1, 1, 3]]
# # b = [4, 7, 3]
# # print(x)
# #
# # # loop run for m times depending on m the error value
# # for i in range(0, 25):
# #     x = seidel(a, x, b)
# #     # print each time the updated solution
# #     print(x)
# #
# # # test_solving_function(seidel_method)