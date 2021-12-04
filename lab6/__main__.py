import numpy as np

def numerically_compute_probability_vec(p, P, eps=0.001):
    """

    :param p: начальный вектор состояния
    :param P: матрица перехода
    :param eps: точность
    :return: стационарный вектор
    """
    while np.abs(np.std(p @ P) - np.std(p)) >= eps:
        p = p @ P

    return p

# начальный вектор состояния
p = np.array([1., 0., 0.])

# матрица перехода
P = np.array([
    [0.2, 0.6, 0.2],
    [0.4, 0.6, 0],
    [0, 0.5, 0.5]
])


pi = numerically_compute_probability_vec(p, P, 0.0001)
print(pi)
