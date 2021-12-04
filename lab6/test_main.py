import numpy as np
from main import numerically_compute_probability_vec, analytically_compute_probability_vec


def test_analytically_compute_probability_vec():
    transition_matrix = np.array([
        [0.2, 0.6, 0.2],
        [0.4, 0.6, 0.0],
        [0.0, 0.5, 0.5],
    ])

    expected_prob_vec = [5 / 17, 10 / 17, 2 / 17]

    actual_prob_vec = analytically_compute_probability_vec(transition_matrix)

    assert np.allclose(actual_prob_vec, expected_prob_vec)


def test_analytically_compute_probability_vec():
    # начальный вектор состояния
    p = np.array([1., 0., 0.])

    # матрица перехода
    P = np.array([
        [0.2, 0.6, 0.2],
        [0.4, 0.6, 0],
        [0, 0.5, 0.5]
    ])

    expected_prob_vec = [5 / 17, 10 / 17, 2 / 17]
    actual_prob_vec = numerically_compute_probability_vec(p, P, eps=0.0001)

    assert np.allclose(actual_prob_vec, expected_prob_vec)
