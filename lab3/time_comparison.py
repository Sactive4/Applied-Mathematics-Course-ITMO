from itertools import count
from timeit import timeit

from scipy.sparse import linalg

from primathlab3 import direct, iteration_method
from primathlab3.math_util import generate_big_matrix, random_vector


def gen_nonsingular_matrix(n, p):
    matrix = generate_big_matrix(n, p, "lil")
    for i in range(n):
        matrix[i, i] += 1000
    return matrix


def gen_test_data(n, p=0.3):
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
        n = 10 ** exp
        matrix, vector = gen_test_data(n, p=0.01)
        exe_time = test_exe_time(solver, matrix, vector)
        print(n, exe_time, sep=", ")
