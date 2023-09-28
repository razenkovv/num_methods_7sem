import numpy as np
from scipy.sparse import csr_array


def create_matrix(n: int, lft=0.0, rgt=1.0):  # -d^2/dx^2 u = lambda * u
    data = np.zeros((n - 2) * 3 + 4)
    col_index = np.zeros_like(data, dtype=int)
    row_count = np.zeros(n + 1, dtype=int)
    data[0] = 2.0; data[1] = -1.0; data[-1] = 2.0; data[-2] = -1.0
    col_index[0] = 0; col_index[1] = 1; col_index[-1] = n-1; col_index[-2] = n-2
    row_count[0] = 0; row_count[1] = 2; row_count[-1] = (n - 2) * 3 + 4

    k = 0
    storage = [-1, 2, -1]
    for i in range(2, (n - 2) * 3 + 2):
        data[i] = storage[k]
        col_index[i] = k + (i - 2) / 3
        k += 1
        k %= 3

    k = 5
    for i in range(2, n):
        row_count[i] = k
        k += 3

    h = (rgt - lft) / n
    return csr_array((data, col_index, row_count), shape=(n, n)) / h**2
