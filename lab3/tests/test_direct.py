import numpy as np
from primathlab3.direct import (
    lower_trivial_system_solution,
    upper_trivial_system_solution,
)
from scipy.sparse import csr_matrix


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
