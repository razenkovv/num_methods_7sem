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

u = t.solve()
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

print("max_diff_with_scipy: ", np.max(np.abs(u - u_sc)))
exact = t.exact(t.xaxis, t.yaxis, delta2)
print("max_scipy_relative_error: ", np.max(np.abs((u_sc - exact) / exact)))
print("max_relative_error: ", np.max(np.abs((u - exact) / exact)))

# p.animation()




# # test
# a = np.array([[1, 0.5, 0.3],
#              [0.5, 0.8, -0.1],
#              [1, 2, -0.75]])
# b = np.array([1,2,3])
# u_scipy = solve(a, b)
# print(u_scipy)
#
# curr_eps = 1
# eps = 1e-8
# rng = np.random.default_rng()
# u = rng.random(3)
# while curr_eps > eps:
#     rk = a @ u - b
#     curr_eps = np.sqrt(rk @ rk)
#     tmp = a @ rk
#     tau = tmp @ rk / (tmp @ tmp)
#     u = u - tau * rk
#
# u_my = u
# print(u_my)
# print(u_scipy)
