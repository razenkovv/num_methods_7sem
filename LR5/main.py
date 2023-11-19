import task as t
import plot as p

u = t.solve()
delta = t.exact(t.xaxis[0], t.yaxis[0], 0) - u[0, 0]

p.plot_exact("jet", "exact_sol.png", delta)
p.plot_step("result", "jet", "calculated.png")
# p.animation()
