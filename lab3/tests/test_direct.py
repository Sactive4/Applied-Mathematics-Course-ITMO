import numpy as np
from primathlab3.direct import (
    lower_trivial_system_solution,
    lu_decomposition,
    upper_trivial_system_solution,
)
from scipy.sparse import csr_matrix


def test_lu_decomposition():
    matrix = csr_matrix(
        np.array(
            [
                [3, 4, -9, 5],
                [-15, -12, 50, -16],
                [-27, -36, 73, 8],
                [9, 12, -10, -16],
            ]
        )
    )

    lower_expected = np.array(
        [
            [1, 0, 0, 0],
            [-5, 1, 0, 0],
            [-9, 0, 1, 0],
            [3, 0, -17 / 8, 1],
        ]
    )

    upper_expected = np.array(
        [
            [3, 4, -9, 5],
            [0, 8, 5, 9],
            [0, 0, -8, 53],
            [0, 0, 0, 653 / 8],
        ]
    )

    lower_actual, upper_actual = lu_decomposition(matrix)
    lower_actual = lower_actual.toarray()
    upper_actual = upper_actual.toarray()

    assert np.isclose(lower_actual, lower_expected).all()
    assert np.isclose(upper_actual, upper_expected).all()


def test_upper_trivial_system_solution():
    matrix = csr_matrix(np.array([[1, 2, 1], [0, 1, 2], [0, 0, 1]]))
    vector = np.array([8, 4, 1])
    expected = np.array([3, 2, 1])

    actual = upper_trivial_system_solution(matrix, vector)

    assert np.isclose(actual, expected).all()


def test_lower_trivial_system_solution():
    matrix = csr_matrix(np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]]))
    vector = np.array([1, 3, 9])
    expected = np.array([1, 1, 4])

    actual = lower_trivial_system_solution(matrix, vector)

    assert np.isclose(actual, expected).all()
