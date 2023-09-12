import numpy as np

from matrix import Matrix

rng = np.random.default_rng(seed=1)

n = 3
low = -10
high = 10

#lft = Matrix(np.array([[1, 1, 1], [0.0, 0, 1], [0.0, 0, 5]]))
lft = Matrix(rng.uniform(size=(n, n), low=low, high=high))
rgt = Matrix(rng.uniform(size=(n, 1), low=low, high=high))

lft.print()
rgt.print()
print('----------')
lft.gauss(rgt)
