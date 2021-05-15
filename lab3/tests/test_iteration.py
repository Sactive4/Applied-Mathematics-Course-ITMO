import numpy as np
import pytest
from primathlab3.iteration_method import seidel_method
from scipy.sparse import csr_matrix


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

def test_seidel_method(matrix, vector, expected_answer):
    matrix = csr_matrix(matrix)
    actual_answer = seidel_method(matrix, vector, 10e-3)
    assert np.isclose(actual_answer, expected_answer).all()
