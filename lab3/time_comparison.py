from primathlab3 import direct
from primathlab3.math_util import generate_big_matrix, random_vector
from timeit import timeit
from scipy.sparse import linalg


def gen_nonsingular_matrix(n, p):
    matrix = generate_big_matrix(n, p).tolil()
    for i in range(n):
        matrix[i, i] += 1
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
    print(f'{"n":<5} {"direct_exe_time":<15} {"scipy_exe_time":<15}')

    for exp in range(1, 7):
        n = 10 ** exp
        matrix, vector = gen_test_data(n)
        direct_exe_time = test_exe_time(direct.system_solution, matrix, vector)
        scipy_exe_time = test_exe_time(scipy_solver, matrix.tocsc(), vector)
        print(f'{n:<5} {direct_exe_time:<15.10f} {scipy_exe_time:<15.10f}')
