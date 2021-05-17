import numpy as np
import pytest
from primathlab3.direct import (
    csr_row_iter,
    lower_trivial_system_solution,
    lu_decomposition,
    system_solution,
    upper_trivial_system_solution,
)
from scipy.sparse import csr_matrix


# TODO:
# def test_lil_row_product():
#     pass


def test_csr_row_iter():
    expected_vals = [0, 0, 1, 2, 0, 3, 0, 0]
    matrix = csr_matrix([expected_vals])
    row_gen = csr_row_iter(matrix, 0)
    actual_vals = list(row_gen)
    assert actual_vals == expected_vals


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
    matrix = csr_matrix(np.array([[2, 1, 1], [0, 3, 2], [0, 0, 7]]))
    vector = np.array([9, 8, 7])
    expected = np.array([3, 2, 1])

    actual = upper_trivial_system_solution(matrix, vector)

    assert np.isclose(actual, expected).all()


def test_lower_trivial_system_solution():
    matrix = csr_matrix(np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]]))
    vector = np.array([1, 3, 9])
    expected = np.array([1, 1, 4])

    actual = lower_trivial_system_solution(matrix, vector)

    assert np.isclose(actual, expected).all()


# генерация тестов https://abakbot.ru/online-16/313-gen-matrix-online


@pytest.mark.parametrize(
    ["matrix", "vector", "expected_answer"],
    [
        (
            [[5, 7, 4], [9, 5, 7], [1, 2, 7]],
            [4, 5, 2],
            [63.0 / 235, 67.0 / 235, 39.0 / 235],
        ),
        (
            [[7, 7, 30], [4, 7, 9], [7, 1, 30]],
            [-6, 6, 3],
            [303.0 / 38, -3.0 / 2, -65.0 / 38],
        ),
    ],
)
def test_system_solution(matrix, vector, expected_answer):
    matrix = csr_matrix(matrix)
    actual_answer = system_solution(matrix, vector)
    assert np.isclose(actual_answer, expected_answer).all()
