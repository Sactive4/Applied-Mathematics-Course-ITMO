import numpy as np


def analytically_compute_probability_vec(transition_matrix):
    equations = transition_matrix.transpose()
    equations -= np.identity(equations.shape[0])

    last_equation = np.ones((1, equations.shape[1]))
    equations = np.append(equations, last_equation, axis=0)

    ordinate = np.zeros(equations.shape[0])
    ordinate[-1] = 1

    probability_vec = np.linalg.lstsq(equations, ordinate, rcond=None)[0]

    return probability_vec
