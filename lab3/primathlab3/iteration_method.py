import numpy as np


def seidel_row_vec_product(csr, vec1, vec2, row_idx):
    start_idx = csr.indptr[row_idx]
    end_idx = csr.indptr[row_idx + 1] if row_idx + 1 < csr.shape[0] else len(csr.data)

    curr_vec = vec1
    product = 0
    diag_value = 0

    for i in range(start_idx, end_idx):
        col_idx = csr.indices[i]
        val = csr.data[i]

        if col_idx >= row_idx:
            curr_vec = vec2

        if col_idx == row_idx:
            diag_value = val
            continue

        product += val * curr_vec[col_idx]

    return product, diag_value


def seidel_method(A, b, eps=1e-6, max_iter=250):
    n = A.shape[0]
    x = np.array(b)

    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            product, diag_value = seidel_row_vec_product(A, x_new, x, i)
            x_new[i] = (b[i] - product) / diag_value

        if np.allclose(x, x_new, rtol=eps):
            break

        x = x_new
    else:
        raise RuntimeError("Seidel method diverges")

    return x
