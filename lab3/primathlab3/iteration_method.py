import numpy as np


def seidel_method(A, b, eps, max_iter=250):
    n = A.shape[0]
    x = np.array(b)

    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, x_new, rtol=eps):
            break

        x = x_new

    return x
