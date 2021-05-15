import numpy as np
import pytest
from primathlab3.iteration_method import seidel_method
from scipy.sparse import csr_matrix


@pytest.mark.parametrize(
    ["matrix", "vector", "expected_answer"],
    [
        (
            [[5, 1, 1], [1, 6, 2], [0, 1, 7]],
            [7, 13, 2],
            [1, 2, 0],
        ),
    ],
)
def test_seidel_method(matrix, vector, expected_answer):
    matrix = csr_matrix(matrix)
    actual_answer = seidel_method(matrix, vector, 10e-3)
    assert np.isclose(actual_answer, expected_answer).all()
