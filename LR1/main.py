import numpy as np

from matrix import Matrix
from methods import Solver

rng = np.random.default_rng(seed=2)

n = 3
low = -10
high = 10

_lft = rng.uniform(size=(n, n), low=low, high=high)
_rgt = np.ravel(rng.uniform(size=(1, n), low=low, high=high))

_lft += np.identity(n) * rng.choice([high, low]) * n  # for diagonal dominance

lft = Matrix(_lft)
rgt = Matrix(_rgt)

# lft.print()
# rgt.print()

solver = Solver()

solver.scipy_routine(lft, rgt)
#solver.gauss_routine(lft, rgt)
solver.jacobi_routine(lft, rgt)
solver.zeidel_routine(lft, rgt)
