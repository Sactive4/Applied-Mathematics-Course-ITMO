import numpy as np
from primathlab3.generators import generate_hilbert_matrix


def test_generate_hilbert_matrix():
    expected = np.array(
        [
            [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5],
            [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6],
            [1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7],
            [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8],
            [1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9],
        ]
    )
    actual = generate_hilbert_matrix(5).toarray()
    assert np.allclose(expected, actual)
