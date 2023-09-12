import numpy as np

from matrix import Matrix

rng = np.random.default_rng(seed=1)

n = 10000
low = -100000000
high = 100000000

_lft = rng.uniform(size=(n, n), low=low, high=high)
_rgt = np.ravel(rng.uniform(size=(1, n), low=low, high=high))

lft = Matrix(_lft)
rgt = Matrix(_rgt)

# lft.print()
# rgt.print()
gauss_result = lft.gauss(rgt)
numpy_solver_result = np.linalg.solve(_lft, _rgt)
print('\nGauss result       : ', gauss_result)
print('Numpy_solver result: ', numpy_solver_result)
print('Error: ', Matrix(gauss_result - np.linalg.solve(_lft, _rgt)).norm())
print('Residual of Gauss: ', (rgt - lft.mult(Matrix(gauss_result))).norm())
print('Residual of Numpy_solver: ', np.linalg.norm(_rgt - _lft @ numpy_solver_result))
