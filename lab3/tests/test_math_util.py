import numpy as np
from primathlab3.math_util import empty_matrix


def test_empty_matrix():
    assert (empty_matrix(2, 3) == np.array([[0, 0, 0], [0, 0, 0]])).all()
