import numpy as np
from scipy.linalg import norm
from matplotlib import pyplot as plt
import time

import potential_box
import eig

n = 1000000  # amount of grid nodes
lft = 0.0  # left edge
rgt = 1.0  # right edge
A = potential_box.create_matrix(n, lft, rgt)

mu0 = 900.0
m = 9
# eigenvector estimate parameter: y0, by default just np.ones() array
t1 = time.time()
y, mu, count1 = eig.inverse_iteration(A, eps=1e-3, mu0=mu0)
y, mu, count2 = eig.rayleigh_iteration(A, eps=1e-6, mu0=mu, y0=y)
t2 = time.time()

mu_exact = (np.pi * m / (rgt - lft))**2
x = np.linspace(lft, rgt, n)
y_exact = np.sin(np.pi*m*x) / norm(np.sin(np.pi*m*x))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y, 'r', ls=':', label='calc', linewidth=5, zorder=2)
ax.plot(x, y_exact, 'g', label='exact', linewidth=2, zorder=1)
fig.suptitle(f'Inverse + Rayleigh iteration; mu0={mu0}')
ax.set_title(f'mu_calc = {mu:.5f}, mu_exact = {mu_exact:.5f} \n {count1} inverse iterations and {count2} Rayleigh iterations\n {n} grid nodes; time: {t2 - t1:.2f} sec')
plt.legend()
plt.show()
