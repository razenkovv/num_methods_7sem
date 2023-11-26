import os
import shutil
from scipy.sparse import csr_array, eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import numpy as np

import task as t
import plot as p

if os.path.exists('data/'):
    shutil.rmtree('data/')
os.mkdir('data/')

u = t.solve(eps=1e-3)
print("\nscipy is solving.")
u_sc = t.solve_scipy()
print("\nscipy solved.")
delta1 = u_sc[0, 0] - u[0, 0]
u_sc -= delta1
delta2 = t.exact(t.xaxis[0], t.yaxis[0], 0) - u[0, 0]
print("delta_with_scipy: ", delta1)
print("delta_with_exact: ", delta2)

p.plot_exact("jet", "exact_sol.png", delta2)
p.plot_step("result", "jet", "calculated.png")
p.plot_smth("jet", "calculated_scipy.png", u_sc, "scipy_calc.")

p.plot_relative_error("jet", "error.png", delta2, u, "error.")
p.plot_relative_error("jet", "error_scipy.png", delta2, u_sc, "scipy_error.")
p.plot_relative_diff_my_scipy("jet", "diff_my_scipy.png", u, u_sc)

print("max_relative_diff_with_scipy: ", np.max(np.abs((u - u_sc) / u_sc)))
xv, yv = np.meshgrid(t.xaxis, t.yaxis)
exact = t.exact(xv, yv, delta2)
print("max_scipy_relative_error: ", np.max(np.abs((u_sc - exact) / exact)))
print("max_relative_error: ", np.max(np.abs((u - exact) / exact)))

# p.animation()
