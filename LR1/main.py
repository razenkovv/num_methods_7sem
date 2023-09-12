import numpy as np

from matrix import Matrix

rng = np.random.default_rng(seed=1)

n = 5
low = -1000
high = 1000

_lft = rng.uniform(size=(n, n), low=low, high=high)
_rgt = np.ravel(rng.uniform(size=(1, n), low=low, high=high))

lft = Matrix(_lft)
rgt = Matrix(_rgt)

lft.print()
rgt.print()
res = lft.gauss(rgt)
print('\nGauss result       : ', res)
print('Numpy_solver result: ', np.linalg.solve(_lft, _rgt))
