import os
import shutil

import task as t
import plot as p

if os.path.exists('data/'):
    shutil.rmtree('data/')
    os.mkdir('data/')

# u = t.solve()
u = t.solve_scipy()

delta = t.exact(t.xaxis[0], t.yaxis[0], 0) - u[0, 0]
print(delta)

p.plot_exact("jet", "exact_sol.png", delta)
p.plot_step("result", "jet", "calculated.png")
p.plot_error("jet", "error.png", delta, u)
# p.animation()

# a, b = t.create_matrix()
# print(a.toarray().astype(int))
