import numpy as np
from scipy import sparse
from scipy.linalg import norm


def inverse_iteration(a: sparse.csr_array, eps: np.float = 1e-4, mu0: np.float = np.pi**2, y0: np.array = None):
    b = a - mu0 * sparse.identity(a.shape[0], format='csr')
    if y0 is None:
        y0 = np.ones(a.shape[0])
    x = y0
    mu = 10 * eps; mu_n = 0.0
    count = 0
    while np.abs(mu_n - mu) > eps:
        x = sparse.linalg.spsolve(b, x)
        x /= norm(x)
        mu = mu_n
        mu_n = a @ x @ x
        count += 1
    return x, mu_n, count


def rayleigh_iteration(a: sparse.csr_array, eps: np.float = 1e-4, mu0: np.float = np.pi**2, y0: np.array = None):
    b = a - mu0 * sparse.identity(a.shape[0], format='csr')
    if y0 is None:
        y0 = np.ones(a.shape[0])
    x = y0
    mu = 10 * eps; mu_n = 0.0
    count = 0
    while np.abs(mu_n - mu) > eps:
        x = sparse.linalg.spsolve(b, x)
        x /= norm(x)
        mu = mu_n
        mu_n = a @ x @ x
        b = a - mu_n * sparse.identity(a.shape[0], format='csr')
        count += 1
    return x, mu_n, count
