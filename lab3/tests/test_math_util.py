import numpy as np
from primathlab3.math_util import empty_matrix, identity_matrix


def test_empty_matrix():
    actual = empty_matrix(2, 3, "csr")
    expected = np.array([[0, 0, 0], [0, 0, 0]])

    assert (actual == expected).all()
    assert actual.getformat() == "csr"


def test_identity_matrix():
    actual = identity_matrix(3, "csr")
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert (actual == expected).all()
    assert actual.getformat() == "csr"
    
