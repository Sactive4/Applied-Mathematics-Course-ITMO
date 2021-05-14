import numpy as np
import pytest
from primathlab3.inverse import inverse_matrix
from scipy.sparse import csr_matrix

# TODO:
# def test_system_solution_matrix():
#     ...


# генерация тестов https://abakbot.ru/online-16/313-gen-matrix-online


@pytest.mark.parametrize(
    ["matrix", "expected_inverse"],
    [
        (
            [[2, 5, 7], [6, 3, 4], [5, -2, -3]],
            [[1, -1, 1], [-38, 41, -34], [27, -29, 24]],
        ),
        (
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 1.0, 0.0]],
            [
                [-1.0 / 8, 3.0 / 8, -1.0 / 2],
                [0.0, 0.0, 1.0],
                [3.0 / 8, -1.0 / 8, -1.0 / 2],
            ],
        ),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_inverse_matrix(matrix, expected_inverse):
    matrix = csr_matrix(matrix)
    actual_inverse = inverse_matrix(matrix).toarray()
    assert np.isclose(actual_inverse, expected_inverse).all()
